function eigval = PTK_GPUVectorisedEigenvalues(M)
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
    
    M1 = gpuArray(M(1,:));
    M2 = gpuArray(M(2,:));
    M3 = gpuArray(M(3,:));
    M4 = gpuArray(M(4,:));
    M5 = gpuArray(M(5,:));
    M6 = gpuArray(M(6,:));
    
    eigval = gather(inner_calculate(M1,M2,M3,M4,M5,M6));
end

function eigval_out = inner_calculate(M1,M2,M3,M4,M5,M6)
%%  GPU function to calculate eigenvalues

numvoxels = size(M1, 2);
eigval = zeros(3, numvoxels, 'single','gpuArray');

m = (M1 + M4 + M6)/3;

q =  (( M1 - m) .* (M4 - m) .* (M6 - m) + ...
    2 * M2 .* M5 .* M3 - ...
    M3.^2 .* (M4 - m) - ...
    M5.^2 .* (M1 - m) - M2.^2 .* (M6 - m) )/2;

p = ( ( M1 - m ).^2 + 2 * M2.^2 + 2 * M3.^2 + ...
    ( M4 - m ).^2 + 2 * M5.^2 + ( M6 - m ).^2 )/6;

p = max(0.01, p);

phi     = 1/3*acos(complex(q./p.^(3/2)));
phi_idx = phi<0;
phi(phi_idx) = phi(phi_idx) + pi/3;

% phi(phi<0) = phi(phi<0) + pi/3;

eigval(1,:) = abs(m + 2*sqrt(p).*cos(phi));
eigval(2,:) = abs(m - sqrt(p).*(cos(phi) + sqrt(3).*sin(phi)));
eigval(3,:) = abs(m - sqrt(p).*(cos(phi) - sqrt(3).*sin(phi)));

[~, i] = sort(abs(eigval));
i = i + size(eigval,1)*ones(size(eigval,1), 1)*(0:size(eigval, 2) - 1);
eigval_out = eigval(i);
end
