classdef PTK_GPULungMaskForRegistration < PTKPlugin
    % PTK_GPULungMaskForRegistration Plugin for finding a mask for the left
    %           or right lung suitable for registration processed on a GPU.
    %
    %     This is a plugin for the Pulmonary Toolkit. Plugins can be run using 
    %     the gui, or through the interfaces provided by the Pulmonary Toolkit.
    %     See PTKPlugin.m for more information on how to run plugins.
    %
    %     This plugin uses library functions edited for GPU usage. Please 
    %     use PTKLungMaskForRegistration otherwise.
    %
    %     Plugins should not be run directly from your code.
    %
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. https://github.com/tomdoel/pulmonarytoolkit
    %     Author: Tom Doel, 2014.  www.tomdoel.com
    %     Editing Author: Felix Bragman, 2015.
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %

    properties
        ButtonText = 'GPU Lung mask<br>for registration'
        ToolTip = ''
        Category = 'GPU'
        AllowResultsToBeCached = true
        AlwaysRunPlugin = true
        PluginType = 'ReplaceOverlay'
        HidePluginInDisplay = false
        FlattenPreviewImage = false
        PTKVersion = '2'
        ButtonWidth = 6
        ButtonHeight = 2
        GeneratePreview = true
        Context = PTKContextSet.SingleLung
        Visibility = 'Developer'
    end
    
    methods (Static)
        
        function results = RunPlugin(dataset, context, reporting)
            
            numDevice = gpuDeviceCount;
            
            if numDevice > 0
            
                results = dataset.GetResult('PTK_GPULeftAndRightLungs', PTKContext.LungROI);
                region = dataset.GetResult('PTKLungRegion', context);
                results.ResizeToMatch(region);
                if context == PTKContext.LeftLung
                    lung_colour = 2;
                elseif context == PTKContext.RightLung
                    lung_colour = 1;
                else
                    reporting.Error('PTKLungMask:InvalidContext', 'PTKLungMask can only be called with the LeftLung or RightLung context');
                end
                results.ChangeRawImage(results.RawImage == lung_colour);
                results.AddBorder(10);
                results = PTK_GPUFillCoronalHoles(results, [], reporting);
                results.RemoveBorder(10);
            
            else
                
                reporting.ShowMessage('No GPU device found, Consider using PTKLungMaskForRegistration');
                
            end
            
            
        end 
    end    
end

