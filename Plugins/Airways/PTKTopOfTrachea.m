classdef PTKTopOfTrachea < PTKPlugin
    % PTKTopOfTrachea. Plugin for finding a point near the top of the trachea.
    %
    %     This is a plugin for the Pulmonary Toolkit. Plugins can be run using 
    %     the gui, or through the interfaces provided by the Pulmonary Toolkit.
    %     See PTKPlugin.m for more information on how to run plugins.
    %
    %     Plugins should not be run directly from your code.
    %
    %     PTKTopOfTrachea runs the library function PTKFindTopOfTrachea to find
    %     the coordinates of a point near the top of the trachea. The
    %     coordinates are stored s global values, relative to the original
    %     image. This plugin also stores global linear indices for other points 
    %     which the algorithm determined are in the trachea. Note that these
    %     points are only a subset of the true set of points within the trachea.
    %     
    %     The output image generated by GenerateImageFromResults changes the
    %     currently displayed image slice in the viewer to show the trachea
    %     point. The point is coloured in red with a red box drawn around it in
    %     all orientations. The other trachea voxels are coloured green.
    %
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. https://github.com/tomdoel/pulmonarytoolkit
    %     Author: Tom Doel, 2012.  www.tomdoel.com
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %
    
    properties
        ButtonText = 'Trachea'
        ToolTip = 'Locates an approximate point for the top of the trachea'
        Category = 'Airways'

        AllowResultsToBeCached = true
        AlwaysRunPlugin = false
        PluginType = 'ReplaceOverlay'
        HidePluginInDisplay = false
        FlattenPreviewImage = true
        PTKVersion = '1'
        ButtonWidth = 6
        ButtonHeight = 2
        GeneratePreview = true
        Visibility = 'Developer'
    end
    
    methods (Static)
        function results = RunPlugin(dataset, reporting)
            reporting.ShowProgress('Finding the top of the trachea');
            if dataset.IsGasMRI
                threshold_image = dataset.GetResult('PTKThresholdGasMRIAirways');
            elseif strcmp(dataset.GetImageInfo.Modality, 'MR')
                lung_threshold = dataset.GetResult('PTKMRILungThreshold');
                threshold_image = lung_threshold.LungMask;
            else
                threshold_image = dataset.GetResult('PTKThresholdLung');
            end            
            [top_of_trachea, trachea_voxels] = PTKFindTopOfTrachea(threshold_image, reporting, PTKSoftwareInfo.GraphicalDebugMode);
            results = [];
            results.top_of_trachea = top_of_trachea;
            results.trachea_voxels = trachea_voxels;
        end
        
        function results = GenerateImageFromResults(trachea_results, image_templates, reporting)
            template_image = image_templates.GetTemplateImage(PTKContext.LungROI);

            top_of_trachea_global = trachea_results.top_of_trachea;
            
            % Convert to local coordinaes relative to the ROI
            top_of_trachea = template_image.GlobalToLocalCoordinates(top_of_trachea_global);
            image_size = template_image.ImageSize;
            
            trachea = zeros(image_size, 'uint8');
            
            trachea_voxels_global = trachea_results.trachea_voxels;
            trachea_voxels_local = template_image.GlobalToLocalIndices(trachea_voxels_global);
            trachea(trachea_voxels_local) = 2;

            trachea(top_of_trachea(1), top_of_trachea(2), top_of_trachea(3)) = 3;
            trachea = PTKImageUtilities.DrawBoxAround(trachea, top_of_trachea, 5, 3);
            
            
            results = template_image;
            results.ChangeRawImage(trachea);
            results.ImageType = PTKImageType.Colormap;
            
            reporting.ChangeViewingPosition(top_of_trachea);
        end
    end    
end