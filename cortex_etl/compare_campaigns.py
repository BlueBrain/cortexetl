import pandas as pd
import cortex_etl as c_etl


# Helper for concatenating dfs for multiple campaigns
def concat_dfs_for_multiple_campaigns(analyzers, dataframe_key):

    list_of_dfs = [a.custom[dataframe_key] for a in analyzers]
    concatenated_dfs = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    return concatenated_dfs


# Compare metrics for a sequential series of campaigns (e.g. an optimisation)
def compare_campaigns(a, a_name):
    
    print("\n----- Compare campaigns -----")

    # Run post analysis + optional plotting for additional campaigns
    analyzers = []
    for conf in a.analysis_config.custom['comparison_campaigns']:
        print(conf)

        ma_for_comparison = c_etl.analysis_initial_processing(conf, loglevel="ERROR")
        a_for_comparison = ma_for_comparison.analyzers[a_name]

        c_etl.post_analysis(a_for_comparison)
        if (a.analysis_config.custom['plot_multi_sim_analysis_for_comparison_campaigns']):
            c_etl.plot_multi_sim_analysis(a_for_comparison)      

        analyzers.append(a_for_comparison)
    analyzers.append(a)


    # Add additional columns with campaign information
    for analyzer_ind, analyzer in enumerate(analyzers):
        analyzer.custom['custom_simulations_post_analysis']['campaign_index'] = analyzer_ind
        analyzer.custom['custom_simulations_post_analysis']['campaign_short_name'] = analyzer.analysis_config.custom['campaign_short_name']


    # Concatenate campagin dfs
    concatenated_custom_simulations_post_analysis = c_etl.concat_dfs_for_multiple_campaigns(analyzers, 'custom_simulations_post_analysis')

    # Lineplots
    hor_key="ca"; ver_key="none"; x_key="desired_connected_proportion_of_invivo_frs"; colour_var_key="campaign_short_name";
    hm_dims = (hor_key, ver_key, x_key, colour_var_key)
    file_dir = str(a.figpaths.campaign_comparison) + '/'
    c_etl.comparison_lineplots(concatenated_custom_simulations_post_analysis, file_dir, *hm_dims)

    # Videos showing how figures change over the campaigns
    c_etl.video_from_image_files([str(a.figpaths.root) + "/ProportionOfInVivo_NonBursting.png" for a in analyzers], str(a.figpaths.campaign_comparison) + '/ProportionOfInVivo_NonBursting.mp4') 
    c_etl.video_from_image_files([str(a.figpaths.root) + "/SingProportionOfInVivo_NonBursting.png" for a in analyzers], str(a.figpaths.campaign_comparison) + '/SingProportionOfInVivo_NonBursting.mp4') 
    c_etl.video_from_image_files([str(a.figpaths.root) + "/FRs_NonBursting.png" for a in analyzers], str(a.figpaths.campaign_comparison) + '/FRs_NonBursting.mp4') 
    c_etl.video_from_image_files([str(a.figpaths.root) + "/DepolM_NonBursting.png" for a in analyzers], str(a.figpaths.campaign_comparison) + '/DepolM_NonBursting.mp4') 
    c_etl.video_from_image_files([str(a.figpaths.root) + "/DesiredUnconnectedFRs.png" for a in analyzers], str(a.figpaths.campaign_comparison) + '/DesiredUnconnectedFRs.mp4') 
