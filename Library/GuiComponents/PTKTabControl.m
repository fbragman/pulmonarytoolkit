classdef PTKTabControl < PTKMultiPanel
    % PTKTabControl. Part of the gui for the Pulmonary Toolkit.
    %
    %     This class is used internally within the Pulmonary Toolkit to help
    %     build the user interface.
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. https://github.com/tomdoel/pulmonarytoolkit
    %     Author: Tom Doel, 2014.  www.tomdoel.com
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %    
    

    properties (Access = protected)
        TabPanel
    end
    
    methods
        function obj = PTKTabControl(parent, reporting)
            obj = obj@PTKMultiPanel(parent, reporting);
            
            obj.TabPanel = PTKTabPanel(obj, reporting);
            obj.AddChild(obj.TabPanel, reporting);
        end
        
        function AddTabbedPanel(obj, panel, name, tag, tooltip)
            % Adds a panel and corresponding tab
            
            obj.TabPanel.AddTab(name, tag, tooltip);
            obj.AddPanel(panel, tag);
            
            % Ensure tab panel will be created last
            obj.Reorder;
        end
        
        function Resize(obj, position)
            
            % Call the PTKPanel superclass, not the PTKMultiPanel, since we are reducing the
            % height to fit in the tab panel
            Resize@PTKPanel(obj, position);
            
            % The inner position takes into account any borders
            inner_position = obj.InnerPosition;

            tab_panel_height = obj.TabPanel.GetRequestedHeight;
            main_panel_height = inner_position(4) - tab_panel_height;
            tab_panel_y_position = inner_position(2) + main_panel_height;
            
            tab_panel_position = [inner_position(1), tab_panel_y_position, inner_position(3), tab_panel_height];
            panel_position = [inner_position(1), inner_position(2), inner_position(3), main_panel_height];

            obj.TabPanel.Resize(tab_panel_position);
            
            % ToDo: We should only need to resize the current tab
            for panel = obj.PanelMap.values
                panel{1}.Resize(panel_position);
            end
        end
        
        function Reorder(obj)
            % Ensures the tab panel is on top
            children = obj.Children;
            tab_panel = obj.TabPanel;
            other_children = {};
            for child = children
                if child{1} ~= tab_panel
                    other_children{end + 1} = child{1};
                end
            end
            
            other_children{end + 1} = tab_panel;
            obj.Children = other_children;
        end
    end
end