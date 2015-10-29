classdef PTK_GPUVesselness < PTKPlugin
    % PTKVesselness. Plugin for detecting blood vessels
    %
    %     This is a plugin for the Pulmonary Toolkit. Plugins can be run using 
    %     the gui, or through the interfaces provided by the Pulmonary Toolkit.
    %     See PTKPlugin.m for more information on how to run plugins.
    %
    %     Plugins should not be run directly from your code.
    %
    %     PTKVesselness computes a mutiscale vesselness filter based on Frangi et
    %     al., 1998. "Multiscale Vessel Enhancement Filtering". The filter
    %     returns a value at each point which in some sense representes the
    %     probability of that point belonging to a blood vessel.
    %
    %     To reduce memory usage, the left and right lungs are filtered
    %     separately and each is further divided into subimages using the
    %     PTKImageDividerHessian function. This will compute the eigenvalues of
    %     the Hessian matrix for each subimage and use these to compute the
    %     vesselness using the PTKComputeVesselnessFromHessianeigenvalues
    %     function.
    %
    %
    %     %%%% COULD TRY AND IMPLEMENT FILTER/MASK IN DIVIDER TO OPERATE
    %     %%%% ON FULL IMAGE INSTEAD OF OCTANTS
    %
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. https://github.com/tomdoel/pulmonarytoolkit
    %     Author: Tom Doel, 2012.  www.tomdoel.com
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %

    properties
        ButtonText = 'GPU Vesselness'
        ToolTip = 'Shows the multiscale vesselness filter for detecting blood vessels'
        Category = 'GPU'
        AllowResultsToBeCached = true
        AlwaysRunPlugin = true
        PluginType = 'ReplaceOverlay'
        HidePluginInDisplay = false
        FlattenPreviewImage = false
        PTKVersion = '1'
        ButtonWidth = 6
        ButtonHeight = 2
        GeneratePreview = true
        Visibility = 'Developer'
    end
    
    methods (Static)
        
        function results = RunPlugin(dataset, reporting)
            tic
            left_and_right_lungs = dataset.GetResult('PTK_GPULeftAndRightLungs');
           
            right_lung = dataset.GetResult('PTK_GPUGetRightLungROI');        
            reporting.PushProgress;
            
            reporting.UpdateProgressStage(0, 2);
            vesselness_right = PTK_GPUVesselness.ComputeVesselness(right_lung,...
                                                                    reporting,...
                                                                       false);
            
            reporting.UpdateProgressStage(1, 2);
            left_lung = dataset.GetResult('PTK_GPUGetLeftLungROI');
            vesselness_left = PTK_GPUVesselness.ComputeVesselness(left_lung, reporting, true);
            
            reporting.PopProgress;

            results = PTKCombineLeftAndRightImages(dataset.GetTemplateImage(PTKContext.LungROI), vesselness_left, vesselness_right, left_and_right_lungs);
            
            lung = dataset.GetResult('PTK_GPULeftAndRightLungs');
            results.ChangeRawImage(results.RawImage.*single(lung.RawImage > 0));
            results.ImageType = PTKImageType.Scaled;
            toc
        end
        
        function results = GenerateImageFromResults(results, ~, ~)
            vesselness_raw = 3*uint8(results.RawImage > 5);
            results.ChangeRawImage(vesselness_raw);
            results.ImageType = PTKImageType.Colormap;
        end        
        
    end
    
    methods (Static, Access = private)
        
        function vesselness = ComputeVesselness(image_data, reporting, is_left_lung)
            
            reporting.PushProgress;
            
            sigma_range = 0.5 : 0.5: 2;
            num_calculations = numel(sigma_range);
            vesselness = [];
            progress_index = 0;
            for sigma = sigma_range
                reporting.UpdateProgressStage(progress_index, num_calculations);
                progress_index = progress_index + 1;
                
                mask = [];
                vesselness_next = PTK_GPUImageDividerHessian(image_data.Copy, @PTK_GPUVesselness.ComputeVesselnessPartImage, mask, sigma, [], false, false, is_left_lung, reporting);
                vesselness_next.ChangeRawImage(100*vesselness_next.RawImage);
                if isempty(vesselness)
                    vesselness =  vesselness_next.Copy;
                else
                    vesselness.ChangeRawImage(max(vesselness.RawImage, vesselness_next.RawImage));
                end
            end
            
            reporting.PopProgress;
            
        end
                
        function vesselness_wrapper = ComputeVesselnessPartImage(hessian_eigs_wrapper, voxel_size)
            vesselness_wrapper = PTKComputeVesselnessFromHessianeigenvalues(hessian_eigs_wrapper, voxel_size);
        end
        
    end
end

