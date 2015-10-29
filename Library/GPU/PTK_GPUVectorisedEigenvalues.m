function [eigval,eigvec] = PTK_GPUVectorisedEigenvalues(M,compute_vec)
    % PTK_GPUVectorisedEigenvalues. Computes eigenvalues and eigenvectors 
    %                               for many symmetric matrices on the GPU
    %
    %     PTKVectorisedEigenvalues is similar to Matlab's eigs() function, but can be
    %     used to compute the eigenvalues and eigenvectors for multiple matrices,
    %     which for a large number of points is significantly quicker than using a
    %     for loop. Each input matrix must be symmetric and is represented by a
    %     single row of the input matrix as described below.
    %
    %     The mex function PTKFastEigenvalues is equivalent to function but runs
    %     faster. PTKVectorisedEigenvalues is slower than PTKFastEigenvalues but still
    %     significantly faster than running eigs() in a for loop when a large
    %     number of matrices is involved.
    %
    %
    %         Syntax:
    %             [eigvectors, eigvalues] = PTKFastEigenvalues(M [, eigenvalues_only])
    %
    %         Input:
    %             M is a 6xn matrix. Each column of M represents one 3x3 symmetric matrix as follows
    %
    %                     [V(1) V(2) V(3); V(2) V(4) V(5); V(3) V(5) V(6)]
    % 
    %                 where V is a 6x1 column of M
    % 
    %             eigenvalues_only is an optional parameter which defaults to
    %                 false. Set to true to only calculate eigenvalues and not
    %                 eigenvectors, which reduces the execution time.
    % 
    %          Outputs:
    %              eigenvalues is a 3xn matrix. Each column contains the 3
    %                  eigenvalues of the matrix V described above
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. https://github.com/tomdoel/pulmonarytoolkit
    %     Author: Tom Doel, 2012.  www.tomdoel.com
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %
    
    if isa(M,'gpuArray')    
        M1 = M(1,:);
        M2 = M(2,:);
        M3 = M(3,:);
        M4 = M(4,:);
        M5 = M(5,:);
        M6 = M(6,:);
        clear M;
    else   
        M1 = gpuArray(M(1,:));
        M2 = gpuArray(M(2,:));
        M3 = gpuArray(M(3,:));
        M4 = gpuArray(M(4,:));
        M5 = gpuArray(M(5,:));
        M6 = gpuArray(M(6,:));
        clear M;
    end
    
    [eigval,tmp_eigvec] = inner_calculate(M1,M2,M3,M4,M5,M6,compute_vec);
    clear M1 M2 M3 M4 M5 M6
    eigval = gather(eigval);
    if ~isempty(tmp_eigvec)
        v1 = gather(tmp_eigvec{1});
        v2 = gather(tmp_eigvec{2});
        v3 = gather(tmp_eigvec{3});
        clear eigvec;
        eigvec = zeros(3,3,size(eigval,2),'single');
        eigvec(:,1,:) = v1;
        eigvec(:,2,:) = v2;
        eigvec(:,3,:) = v3; 
    end
end

function [eigval,eigvec] = inner_calculate(M1,M2,M3,M4,M5,M6,compute_vec)
%%  GPU function to calculate eigenvalues

numvoxels = size(M1, 2);
eigval = zeros(3, numvoxels,'single','gpuArray');

m = (M1 + M4 + M6)/3;

q = ( (M1-m).*(M4-m).*(M6-m) + ...
              2*M2 .* M5.*M3 - ...
            M3.*M3 .* (M4-m) - ...
      M5.*M5.*(M1-m) - M2.*M2.*(M6-m) )/2;

p = ( (M1-m).*(M1-m) + 2 * M2.*M2 + 2 * M3.*M3 + ...
      (M4-m).*(M4-m) + 2 * M5.*M5 + (M6-m).*(M6-m) )/6;

p = max(0.01,p);

phi     = 1/3*acos(q./p.^(3/2));
phi(phi<0) = phi(phi<0) + pi/3;

sqr3 = sqrt(3);

eigval(1,:) = m + 2*sqrt(p).*cos(phi);
eigval(2,:) = m - sqrt(p).*(cos(phi) + sqr3.*sin(phi));
eigval(3,:) = m - sqrt(p).*(cos(phi) - sqr3.*sin(phi));

[~, i] = sort(abs(eigval));
i = i + size(eigval,1)*ones(size(eigval,1), 1)*(0:size(eigval, 2) - 1);
eigval = eigval(i);
% Clear GPU variables
clear p q m phi

if compute_vec
    
    eigvec = cell(1,3);
    
    for l = 1 : 2
        Ai = M1 - eigval(l,:);
        Bi = M4 - eigval(l,:);
        Ci = M6 - eigval(l,:);
        
        eix = ( M2.*M5 - Bi.*M3 ) .* (M3 .*M5 - Ci.*M2 );
        eiy = ( M3.*M5 - Ci.*M2 ) .* (M3 .*M2 - Ai.*M5 );
        eiz = ( M2.*M5 - Bi.*M3 ) .* (M3 .*M2 - Ai.*M5 );
        
        vec = sqrt(eix.*eix + eiy.*eiy + eiz.*eiz);      
        vec = max(0.01,vec);
        eigvec{l} = [eix; eiy; eiz] ./ vec([1;1;1], :);
    end

eigvec{3} = cross_product(eigvec{1},eigvec{2});
    
else
    
    eigvec = [];
    
end

end

function out = cross_product(vec1,vec2)
%%  Quicker GPU cross product

out = [ vec1(2,:).*vec2(3,:) - vec1(3,:).*vec2(2,:); ...
        vec1(3,:).*vec2(1,:) - vec1(1,:).*vec2(3,:); ...
        vec1(1,:).*vec2(2,:) - vec1(2,:).*vec2(1,:); ];

% out      = vec1(2,:).*vec2(3,:) - vec1(3,:).*vec2(2,:);
% out(2,:) = vec1(3,:).*vec2(1,:) - vec1(1,:).*vec2(3,:);
% out(3,:) = vec1(1,:).*vec2(2,:) - vec1(2,:).*vec2(1,:);

end