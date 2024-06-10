import numpy
import pandas
import os
import tqdm
import cortexetl as c_etl
from blueetl.parallel import call_by_simulation
import pandas as pd
from functools import partial
from matplotlib import pyplot as plt



def make_t_bins(t_start, t_end, t_step):
    t_bins = numpy.arange(t_start, t_end + t_step, t_step)
    return t_bins

def flatten_locations(locations, flatmap):
    if isinstance(flatmap, list):
        flat_locations = locations[flatmap].values
    else:
        from voxcell import VoxelData
        fm = VoxelData.load_nrrd(flatmap)
        flat_locations = fm.lookup(locations.values).astype(float)
        flat_locations[flat_locations == -1] = numpy.NaN
    return pandas.DataFrame(flat_locations, index=locations.index)


def make_spatial_bins(flat_locations, nbins=1000):
    mn = numpy.nanmin(flat_locations, axis=0)
    mx = numpy.nanmax(flat_locations, axis=0)
    ratio = (mx[1] - mn[1]) / (mx[0] - mn[0]) # ratio * nx ** 2 = nbins
    nx = int(numpy.sqrt(nbins / ratio))
    ny = int(nbins / nx)
    binsx = numpy.linspace(mn[0], mx[0] + 1E-3, nx + 1)
    binsy = numpy.linspace(mn[1], mx[1] + 1E-3, ny + 1)
    return binsx, binsy

def make_histogram_function(t_bins, loc_bins, location_dframe, spikes):
    t_step = numpy.mean(numpy.diff(t_bins))
    fac = 1000.0 / t_step
    nrns_per_bin = numpy.histogram2d(location_dframe.values[:, 0],
                                     location_dframe.values[:, 1],
                                     bins=loc_bins)[0]
    nrns_per_bin = nrns_per_bin.reshape((1,) + nrns_per_bin.shape)

    spikes = spikes.loc[numpy.in1d(spikes.values, location_dframe.index.values)]
    t = spikes.index.values
    loc = location_dframe.loc[spikes['gid']].values
    raw, _ = numpy.histogramdd((t, loc[:, 0], loc[:, 1]), bins=(t_bins,) + loc_bins)
    raw = fac * raw / (nrns_per_bin + 1E-6)
    return raw

def save(Hs, t_bins, loc_bins, out_root):
    if not os.path.isdir(out_root):
        _ = os.makedirs(out_root)
    import h5py
    h5 = h5py.File(os.path.join(out_root, "spiking_activity_3d.h5"), "w")
    grp_bins = h5.create_group("bins")
    grp_bins.create_dataset("t", data=t_bins)
    grp_bins.create_dataset("x", data=loc_bins[0])
    grp_bins.create_dataset("y", data=loc_bins[1])

    grp_data = h5.create_group("histograms")
    for i, val in enumerate(Hs.get()):
        grp_data.create_dataset("instance{0}".format(i), data=val)
    mn_data = numpy.mean(numpy.stack(Hs.get(), -1), axis=-1)
    grp_data.create_dataset("mean", data=mn_data)
    return mn_data

def plot_and_save_single_image(hist, path):

    plt.figure()
    plt.imshow(hist)
    plt.savefig(path)
    plt.close()

import os
import numpy
import tqdm
import matplotlib
def plot(Hs, t_bins, loc_bins, images_dir, delete_images, video_output_root, min_color_lim_pct=-1):
    if not os.path.isdir(images_dir):
        _ = os.makedirs(images_dir)
    from matplotlib import pyplot as plt
    # for bin_index in list(range(len(t_bins))):
    # flattened_Hs = Hs.flatten()
    # print(flattened_Hs.shape)
    # print(numpy.max(flattened_Hs))
    mx_clim = numpy.percentile(Hs, 99)
    mn_clim = 0
    if (min_color_lim_pct != -1):
        mn_clim = numpy.percentile(Hs, 95)
    print(mn_clim)
#     Hs[Hs <= 0.05] = numpy.nan
#     print(Hs)
    
    cmap = matplotlib.cm.cividis
    cmap.set_bad('white',1.)
        
    # mx_clim = numpy.max(Hs)
    # mx_clim = numpy.percentile(Hs[Hs > 0], 90)
    fps = []
    for t_start, t_end, bin_index in tqdm.tqdm(zip(t_bins[:-1], t_bins[1:], list(range(len(t_bins))))):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        img = ax.imshow(Hs[bin_index, :, :], cmap=cmap)
        img.set_clim([mn_clim, mx_clim])
        ax.set_title("{0} - {1} ms".format(t_start, t_end))
        plt.colorbar(img, cmap=cmap, label='FR (spikes / s')
        plt.box(False)
        fn = "frame{:04d}.png".format(bin_index)
        fp = os.path.join(images_dir, fn)
        fig.savefig(fp)
        fps.append(fp)
        if (bin_index == 0):
            fn = "frame{:04d}.pdf".format(bin_index)
            fp = os.path.join(images_dir, fn)
            fig.savefig(fp)
        
        plt.close(fig)

#     print(video_output_root + ".mp4")
    c_etl.video_from_image_files(fps, video_output_root + ".mp4")
    if delete_images:
        for f in fps:
            os.remove(f)

import numpy
from scipy.ndimage import gaussian_filter
def single_flatspace_video(simulation_row, filtered_dataframes, flat_locations, flatspace_video_opt, analysis_config, flatspace_path_pre=None, images_dir=None):

    window_row = filtered_dataframes['windows'].iloc[0]

    if (flatspace_path_pre==None):
        flatspace_path_pre = flatspace_video_opt['video_output_root'] + str(simulation_row['simulation_id']) + "_" + simulation_row['simulation_string']
    if (images_dir==None):
        images_dir = str(window_row['flatspace_video_images_dir']) + "/" + flatspace_video_opt['vid_str'] + "_" + str(simulation_row['simulation_id']) + "/"
    

    t_bins = make_t_bins(window_row['t_start'], window_row['t_stop'], flatspace_video_opt['t_step'])
    spikes = filtered_dataframes['spikes'].loc[:, ['time', 'gid']].set_index('time')
    
    loc_bins = make_spatial_bins(flat_locations, flatspace_video_opt['n_spatial_bins'])
    spatial_temporal_hist = make_histogram_function(t_bins, loc_bins, flat_locations, spikes)
    smoothed_spatial_temporal_hist = gaussian_filter(spatial_temporal_hist, [flatspace_video_opt['temporal_smoothing_sigma'], 1.0, 1.0])

    plot(smoothed_spatial_temporal_hist, t_bins, loc_bins, images_dir, flatspace_video_opt['delete_images'], flatspace_path_pre)

    hist = smoothed_spatial_temporal_hist
    hist_mean = numpy.mean(hist, axis=0)
    plot_and_save_single_image(hist_mean, flatspace_path_pre + '_hist_mean.pdf')

    if (flatspace_video_opt['stim_anal'] != None):

        # where_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= 20.0), ((t_bins) % 500.0) < 100.0)).flatten()
        # where_not_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= 0.0), ((t_bins) % 500.0) >= 100.0)).flatten()

        where_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= flatspace_video_opt['stim_anal']['stim_period'][0]), ((t_bins)) < flatspace_video_opt['stim_anal']['stim_period'][1])).flatten()
        where_not_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= flatspace_video_opt['stim_anal']['spont_period'][0]), ((t_bins)) < flatspace_video_opt['stim_anal']['spont_period'][1])).flatten()

        hist_stim = hist[where_stim[:-1]]
        hist_not_stim = hist[where_not_stim[:-1]]
        hist_stim_mean = numpy.mean(hist_stim, axis=0)
        hist_not_stim_mean = numpy.mean(hist_not_stim, axis=0)

        hist_stim_mean_diff = hist_stim_mean - hist_not_stim_mean
        log_hist_stim_mean_diff = numpy.log(hist_stim_mean_diff)

        stim_minus_spont = hist_stim - hist_not_stim_mean

        plot(stim_minus_spont, t_bins[where_stim], loc_bins, images_dir, flatspace_video_opt['delete_images'], flatspace_path_pre + '_stim_minus_spont')
        plot(stim_minus_spont, t_bins[where_stim], loc_bins, images_dir, flatspace_video_opt['delete_images'], flatspace_path_pre + '_stim_minus_spont_min_lim_60', min_color_lim_pct=60)
        # plot(stim_minus_spont, t_bins[where_stim], loc_bins, images_dir, flatspace_video_opt['delete_images'], flatspace_path_pre + '_stim_minus_spont_min_lim_60_log', min_color_lim_pct=60)
        # plot(numpy.log(hist_stim - hist_not_stim_mean), t_bins[where_stim], loc_bins, images_dir, flatspace_path_pre + 'log_subtrac_mean')

        plot_and_save_single_image(hist_not_stim_mean, flatspace_path_pre + '_hist_not_stim_mean.pdf')
        plot_and_save_single_image(hist_stim_mean, flatspace_path_pre + '_hist_stim_mean.pdf')
        plot_and_save_single_image(hist_stim_mean_diff, flatspace_path_pre + '_hist_stim_mean_diff.pdf')
        plot_and_save_single_image(log_hist_stim_mean_diff, flatspace_path_pre + '_log_hist_stim_mean_diff.pdf')
        plot_and_save_single_image(log_hist_stim_mean_diff, flatspace_path_pre + '_log_hist_stim_mean_diff_-4_-2.pdf')


    r_dict = {"smoothed_spatial_temporal_hist": smoothed_spatial_temporal_hist,
            "t_bins": t_bins}
    return r_dict



import os
from blueetl.parallel import call_by_simulation
from functools import partial
def flatspace_videos(a):

    print("\n----- Flatspace videos -----")
    for flatspace_video_key in a.analysis_config.custom['flatspace_videos']:
        flatspace_video_opt = a.analysis_config.custom['flatspace_videos'][flatspace_video_key]
        flatspace_video_opt['vid_str'] = flatspace_video_opt['window'] + "_" + str(flatspace_video_opt['t_step']) + "_" + str(flatspace_video_opt['n_spatial_bins']) + "_" + str(flatspace_video_opt['temporal_smoothing_sigma'])
        flatspace_video_opt['video_output_root'] = str(a.figpaths.flatspace_videos) + "/" + flatspace_video_opt['vid_str'] + "/"
        os.makedirs(flatspace_video_opt['video_output_root'], exist_ok=True)

        dataframes={
            "circuits": a.repo.simulations.df.loc[:, ['circuit', 'circuit_id', 'simulation_id']],
            "spikes": a.repo.report.df.etl.q(neuron_class="ALL", window=flatspace_video_opt['window']),
            "windows": a.repo.windows.df.etl.q(window=flatspace_video_opt['window']), 
            "neurons": a.repo.neurons.df.etl.q(neuron_class="ALL")}

        gids = a.repo.neurons.df.etl.q(circuit_id=0)['gid']
        locations = a.repo.simulations.df.loc[:, ['circuit', 'circuit_id', 'simulation_id']].iloc[0]['circuit'].cells.get(gids, ["x", "y", "z"])
        flat_locations = c_etl.flatten_locations(locations, a.analysis_config.custom["flatmap"])
        
        results = call_by_simulation(a.repo.simulations.df, 
                                        dataframes, 
                                        func=partial(single_flatspace_video, 
                                                    flat_locations=flat_locations, 
                                                    flatspace_video_opt=flatspace_video_opt, 
                                                    analysis_config=a.analysis_config.custom,
                                                    flatspace_path_pre=None, 
                                                    images_dir=None),
                                        how='series')



#         hist = results[0]['smoothed_spatial_temporal_hist']
#         t_bins = results[0]['t_bins']


#         where_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= 20.0), ((t_bins)) < 60.0)).flatten()
#         where_not_stim = numpy.argwhere(numpy.logical_and(((t_bins) >= 0.0), ((t_bins)) >= 60.0)).flatten()

#         hist_stim = hist[where_stim[:-1]]
#         hist_not_stim = hist[where_not_stim[:-1]]

#         hist_stim_mean = numpy.mean(hist_stim, axis=0)
#         hist_not_stim_mean = numpy.mean(hist_not_stim, axis=0)

#         hist_stim_mean_diff = hist_stim_mean - hist_not_stim_mean
#         log_hist_stim_mean_diff = numpy.log(hist_stim_mean_diff)

#         plot_and_save_single_image(hist_not_stim_mean,  'hist_not_stim_mean.pdf')
#         plot_and_save_single_image(hist_stim_mean,  'hist_stim_mean.pdf')
#         plot_and_save_single_image(hist_stim_mean_diff,  'hist_stim_mean_diff.pdf')
#         plot_and_save_single_image(log_hist_stim_mean_diff,  'log_hist_stim_mean_diff.pdf')
#         plot_and_save_single_image(log_hist_stim_mean_diff,  'log_hist_stim_mean_diff_-4_-2.pdf')




# OTHER

# if cfg["show_inputs"]:
#     from conntility.circuit_models import neuron_groups
#     input_props = neuron_groups.load_all_projection_locations(circ,
#     ["x", "y", "z", "u", "v", "w"] + neuron_groups.SS_COORDINATES)
#     input_props = input_props.set_index(pandas.RangeIndex(len(input_props)))
#     flat_col_names = [neuron_groups.SS_COORDINATES[0], neuron_groups.SS_COORDINATES[2]]
#     flat_locations = input_props.set_index("sgid")[flat_col_names]
# else: