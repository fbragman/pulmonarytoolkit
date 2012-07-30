function TDVisualiseAirwayGrowingTree(airway_tree, reporting)
    % TDVisualiseAirwayGrowingTree. Draws a simplified visualisation of a tree
    % generated by TDAirwayGenerator.
    %
    %     Syntax
    %     ------
    %
    %         TDVisualiseAirwayGrowingTree(parent_branch)
    %
    %             parent_branch     is the root branch in a TDTreeModel structure 
    %             reporting         is an object which implements
    %                               TDReportingInterface, for error and progress
    %
    %     Licence
    %     -------
    %     Part of the TD Pulmonary Toolkit. http://code.google.com/p/pulmonarytoolkit
    %     Author: Tom Doel, 2012.  www.tomdoel.com
    %     Distributed under the GNU GPL v3 licence. Please see website for details.
    %           
    % 

    reporting.ShowProgress('Drawing computed airway tree');
    % Assumes coordinates are in mm so no voxel size conversion required
    aspect_ratio = [1, 1, 1];

    % Set up appropriate figure properties
    fig = figure;
    set(fig, 'Name', 'Airway Growing Tree');
    
    hold on;
    axis off;
    axis square;
    lighting gouraud;
    axis equal;
    set(gcf, 'Color', 'white');
    set(gca, 'DataAspectRatio', aspect_ratio);
    rotate3d
    cm = colormap('lines');
    view(-37.5, 30);
    cl = camlight('headlight');

    segments_to_do = airway_tree;
    segment_count = 0;
    total_segments = airway_tree.CountBranches;
    
    while ~isempty(segments_to_do)
        if reporting.HasBeenCancelled
            return
        end
        segment_count = segment_count + 1;
        reporting.UpdateProgressValue(round(100*segment_count/total_segments));
        segment = segments_to_do(end);
        segments_to_do(end) = [];
        segments_to_do = [segments_to_do segment.Children];
        
        start_point = segment.StartCoords;
        end_point = segment.EndCoords;
        generation = segment.GenerationNumber;
        
        colour_value = mod(generation - 1, 5) + 1;
        colour = cm(colour_value, :);

        if segment.IsGenerated
            thickness = 1;
        else
             colour = [1 1 0];
            thickness = 2;
        end
        
        
        pi = [start_point(1); end_point(1)];
        pj = [start_point(2); end_point(2)];
        pk = [start_point(3); end_point(3)];
        pk = - pk;
        
        
        line('XData', pj, 'YData', pk, 'ZData', pi, 'Color', colour, 'LineWidth', thickness);
    end
end

