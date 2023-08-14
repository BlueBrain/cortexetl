import os
import pandas as pd

def create_figure_dirs(a):

    a.figpaths = type('', (), {})()
    a.figpaths.root = a.analysis_config.output
    if ("sim_filter_in_memory_name" in list(a.analysis_config.custom.keys())):
        a.figpaths.root = a.figpaths.root / a.analysis_config.custom['sim_filter_in_memory_name']
    a.figpaths.root = a.figpaths.root / "figures"
    
    for folder in ['evoked', 'heatmaps', 'lineplots', 'parameter_effects', 'raster_videos', 'fft_videos', 
                   '3x3s', 'fr_condition_comparisons', 'campaign_comparison', 'flatspace_videos', 'multi_hex',
                  'individual_simulations']:
        a.figpaths.__dict__[folder] = a.figpaths.root / folder
        os.makedirs(a.figpaths.__dict__[folder], exist_ok=True)

    temp_sims_df = a.repo.simulations.df.set_index(a.analysis_config.custom["independent_variables"])
    a.repo.simulations.df['simulation_string'] = temp_sims_df.index
    a.repo.simulations.df['simulation_string'] = a.repo.simulations.df['simulation_string'].astype(str)
    a.repo.simulations.df['figures_dir'] = a.figpaths.individual_simulations / a.repo.simulations.df['simulation_string']
    a.repo.simulations.df.apply(lambda row : os.makedirs(row['figures_dir'], exist_ok=True), axis=1)

    a.repo.simulations.df['rasters_dir'] = a.repo.simulations.df['figures_dir'] / "Rasters"
    a.repo.simulations.df['fft_dir'] = a.repo.simulations.df['figures_dir'] / "FFT"
    a.repo.simulations.df['flatspace_video_images_dir'] = a.repo.simulations.df['figures_dir'] / "FlatspaceVideos"
    a.repo.simulations.df.apply(lambda row : os.makedirs(row['rasters_dir'], exist_ok=True), axis=1)
    a.repo.simulations.df.apply(lambda row : os.makedirs(row['fft_dir'], exist_ok=True), axis=1)
    a.repo.simulations.df.apply(lambda row : os.makedirs(row['flatspace_video_images_dir'], exist_ok=True), axis=1)

    merged = pd.merge(a.repo.windows.df, a.repo.simulations.df, how="left")
    a.repo.windows.df["rasters_dir"] = merged["rasters_dir"]
    a.repo.windows.df["flatspace_video_images_dir"] = merged["flatspace_video_images_dir"]
    a.repo.windows.df.apply(lambda row : os.makedirs(row['rasters_dir'], exist_ok=True), axis=1)
    a.repo.windows.df.apply(lambda row : os.makedirs(row['flatspace_video_images_dir'], exist_ok=True), axis=1)

    # USEFUL FOR CREATING WINDOW DIRECTORY WITHIN SIMULATION DIRECTORY
    # a.repo.windows.df['figures_dir'] = windows['figures_dir'] + windows['window'].astype(str) + '/'
    # a.repo.windows.df.apply(lambda row : os.makedirs(row['figures_dir'], exist_ok=True), axis=1)