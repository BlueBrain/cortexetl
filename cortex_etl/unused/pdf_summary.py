from PIL import Image
from fpdf import FPDF


def image_height_and_width(image_path):
    image = Image.open(image_path)
    width, height = image.size

    return width, height


class PDF(object):

    def __init__(self, pdf_h, pdf_w):
        """
        """
        self.pdf_h = pdf_h
        self.pdf_w = pdf_w
        self.pdf = FPDF('L', 'mm', (pdf_h, pdf_w))

    def new_page(self):
        self.pdf.add_page()

    def add_txt(self, xy, s, txt):

        self.pdf.set_xy(*xy); 
        self.pdf.set_font('Arial', 'B', s); 
        self.pdf.cell(w=1, h=1, align='L', txt=txt, border=0)

    def add_image(self, image_path, x=0, y=0, max_h=-1, max_w=-1):

        if (max_h == -1):
            max_h = self.pdf_h
        if (max_w == -1):
            max_w = self.pdf_w

        width, height = image_height_and_width(image_path)

        image_ratio = width/height
        max_ratio = max_w/max_h

        if (image_ratio > max_ratio):
            self.pdf.image(image_path, x=x, y=y, w=max_w)
            return x, y, max_w, max_w/image_ratio
        else:
            self.pdf.image(image_path, x=x, y=y, h=max_h)
            return x, y, max_h*image_ratio, max_h


    def save(self, filename):
        self.pdf.output(filename, 'F')


def multi_sim_summary_pdfs(a):

    file_dir = str(a.analysis_config['heatmaps']) + '/'

    pdf = c_etl.PDF(400, 400)

    pdf.new_page()
    pdf.add_txt((150,3), 12, "ParameterEffectSummary")
    x, y, width, height = pdf.add_image(file_dir + "bursting.png", x=0, y=10)
    x, y, width, height = pdf.add_image(file_dir + "atleast_one_neuron_class_fr_greater_than_invivo_thresh.png", x=0, y=y + height)
    x, y, width, height = pdf.add_image(file_dir + "bursting_or_fr_above_threshold.png", x=0, y=y + height)

    pdf.new_page()
    x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval.png", x=0, y=10)
    x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval_M-bursting_or_fr_above_threshold.png", x=0, y=y + height)
    x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval_M-bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range.png", x=0, y=y + height)

    pdf.new_page()
    x, y, width, height = pdf.add_image(file_dir + "std_of_neuron_class_difference_to_in_vivo_M-bursting_or_fr_above_threshold.png", x=0, y=10)
    x, y, width, height = pdf.add_image(file_dir + "cv_of_neuron_class_difference_to_in_vivo_M-bursting_or_fr_above_threshold.png", x=0, y=y + height)

    pdf.save(file_dir + "MultiSimSummary.pdf")

    


file_dir = str(a.analysis_config['heatmaps']) + '/'

pdf = PDF(400, 400)

pdf.new_page()
pdf.add_txt((150,3), 12, "ParameterEffectSummary")
x, y, width, height = pdf.add_image(file_dir + "bursting.png", x=0, y=10)
x, y, width, height = pdf.add_image(file_dir + "atleast_one_neuron_class_fr_greater_than_invivo_thresh.png", x=0, y=y + height)
x, y, width, height = pdf.add_image(file_dir + "bursting_or_fr_above_threshold.png", x=0, y=y + height)

pdf.new_page()
x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval.png", x=0, y=10)
x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval_M-bursting_or_fr_above_threshold.png", x=0, y=y + height)
x, y, width, height = pdf.add_image(file_dir + "ei_corr_rval_M-bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range.png", x=0, y=y + height)

pdf.new_page()
x, y, width, height = pdf.add_image(file_dir + "std_of_neuron_class_difference_to_in_vivo_M-bursting_or_fr_above_threshold.png", x=0, y=10)
x, y, width, height = pdf.add_image(file_dir + "cv_of_neuron_class_difference_to_in_vivo_M-bursting_or_fr_above_threshold.png", x=0, y=y + height)

pdf.save(file_dir + "MultiSimSummary.pdf")




# def create_individual_sim_summary(simulation_row, simulation_windows, analysis_config):       

#   ######################################################### PDF PARAMETERS
#   spont_time_window_key = "single_spontaneous_window_2000_7000"

#   fpdf_w=400
#   fpdf_h=300
    
#   # y coords
#   main_title_y = 3
#   subtitle_y = main_title_y + 5
#   raster_y = subtitle_y + 3
#   ei_y = raster_y + 52
#   fr_y = ei_y + 30
#   spont_stats_y = ei_y + 30
#   spont_met_y_start = spont_stats_y + 100
#   evoked_stats_y = ei_y + 30
#   evoked_met_y_start = evoked_stats_y + 135

#   # x coords
#   spont_figs_x_start = 0
#   spont_met_x_start = 80  
#   spont_met_x_indent = spont_met_x_start + 1
#   evoked_figs_x_start = 134
#   evoked_stats_x_buffer = 30
#   evoked_met_x_start = evoked_figs_x_start + 40
#   evoked_met_x_indent = evoked_met_x_start + 1
#   second_evoked_figs_x_start = 267

#   # widths
#   raster_width = 130
#   fr_width = 90
#   spont_stats_width = 70
#   evoked_stats_width = 100


#   ######################################################### PDF SETUP
    
#   pdf = c_etl.PDF(fpdf_h, fpdf_w)
#   pdf.new_page()
#   # bnac.pdf_txt(fpdf, (150,main_title_y), 12, sim_row.loc[('simulation', 'paths', 'SimulationString')])
#   pdf.add_txt((150,main_title_y), 12, "Test")


#   # ######################################################### SPONTANEOUS FIGURES

#   # bnac.pdf_txt(fpdf, (raster_width/2, subtitle_y), 10, config['temp_secondary_spont_time_window_key'])
#   # fpdf.image(config['dirs'][config['temp_secondary_spont_time_window_key']]['1stTrialRastersCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRastersCompNGs.png', x=spont_figs_x_start, y=raster_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['temp_secondary_spont_time_window_key']]['1stTrialRasterAllGroupsBigNoSpikesALLEI'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRasterAllGroupsBigNoSpikesALLEI.png', x=spont_figs_x_start, y=ei_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['temp_secondary_spont_time_window_key']]['MultipleStatsCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_COMPMultipleStatsCompNGs.png', x=spont_figs_x_start, y=spont_stats_y, w=spont_stats_width)
#   # fpdf.image(config['dirs'][config['main_spont_comparison_time_window_key']]['SingleStatGrids'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + 'mean_of_mean_firing_rates_per_second.png', x=spont_figs_x_start + spont_stats_width, y=fr_y, w=fr_width)



#   # ######################################################### SPONTANEOUS METRICS

#   # spont_not_bursting = not bool(bnac.final_metric('Spont_Bursting', sim_row, None, config['temp_secondary_spont_time_window_key'], '', "ALL", config, spontaneous_key=config['temp_secondary_spont_time_window_key'])[0])
#   # add_criteria_string(spont_not_bursting, "No spontaneous burst", (spont_met_x_start,spont_met_y_start), fpdf)

#   # for i, ng_key in enumerate(config['comparison_ng_keys']):
#   #   fr_peak_sub1Hz = bool(bnac.final_metric('FRPeakSub1Hz', sim_row, None, config['temp_secondary_spont_time_window_key'], '', ng_key, config, spontaneous_key=config['temp_secondary_spont_time_window_key'])[0])
#   #   add_criteria_string(fr_peak_sub1Hz, ng_key + "- Sub 1Hz peak", (spont_met_x_indent, spont_met_y_start + 20 + (i+1) * 3), fpdf)

#   # # vivo_time_window_key = config["time_window_dicts"][config['main_spont_comparison_time_window_key']]["vivo_time_window_key"]
#   # vivo_row, vivo_time_window_key = bnac.select_vivo_row_for_time_window(config['main_spont_comparison_time_window_key'], vivo_stim_row, vivo_spont_row, config)
#   # print(vivo_time_window_key)
#   # for i, ng_key in enumerate(config['comparison_ng_keys']):
#   #   fr_lessthan_vivo = bool(bnac.final_metric('FRLessThanVivo', sim_row, vivo_row, config['main_spont_comparison_time_window_key'], vivo_time_window_key, ng_key, config, spontaneous_key=config['main_spont_comparison_time_window_key'])[0])
#   #   add_criteria_string(fr_lessthan_vivo, ng_key + "- FR < Vivo", (spont_met_x_indent, spont_met_y_start + 40 + (i+1) * 3), fpdf)

#   # good_ei_correlation_r_val = bool(bnac.final_metric('SmoothedFREICorrelationGoodBinary', sim_row, None, config['temp_secondary_spont_time_window_key'], '', ng_key, config, spontaneous_key=config['temp_secondary_spont_time_window_key'])[0])
#   # add_criteria_string(good_ei_correlation_r_val, "Smoothed E/I correlation r: " + str(bnac.round_sig(sim_row.loc[(config['temp_secondary_spont_time_window_key'], "BetweenGroupStats", "smoothed_ei_fr_correlation_sd3ms_1msbin")], sig=2)), (spont_met_x_indent,spont_met_y_start + 5), fpdf)


#   # bnac.pdf_txt(fpdf, (spont_met_x_start,spont_met_y_start + 10), 10, str(np.min(sim_row.loc[(config['temp_secondary_spont_time_window_key'], "ALL", 'smoothed_3ms_spike_times_max_normalised_hist_1ms_bin')])))
#   # bnac.pdf_txt(fpdf, (spont_met_x_start,spont_met_y_start + 15), 10, str(np.max(sim_row.loc[(config['temp_secondary_spont_time_window_key'], "ALL", 'smoothed_3ms_spike_times_max_normalised_hist_1ms_bin')])))



#   # ######################################################### EVOKED FIGURES

#   # fpdf.set_text_color(0, 0, 0);     
#   # bnac.pdf_txt(fpdf, (raster_width + raster_width/2,subtitle_y), 10, config['main_evoked_comparison_time_window_key'])
#   # fpdf.image(config['dirs'][config['main_evoked_comparison_time_window_key']]['1stTrialRastersCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRastersCompNGs.png', x=evoked_figs_x_start, y=raster_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['main_evoked_comparison_time_window_key']]['1stTrialRasterAllGroupsBigNoSpikesALLEI'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRasterAllGroupsBigNoSpikesALLEI.png', x=evoked_figs_x_start, y=ei_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['main_evoked_comparison_time_window_key']]['MultipleStatsCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_COMPMultipleStatsCompNGs.png', x=evoked_figs_x_start + evoked_stats_x_buffer, y=evoked_stats_y, w=evoked_stats_width)
    
#   # # fpdf.image(config['dirs'][config['time_window_keys_to_process'][1]]['SingleStatGrids'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + 'mean_of_mean_firing_rates_per_second.png', x=155, y=140, w=100)



#   # # ######################################################### EVOKED METRICS

#   # evoked_not_bursting = not bool(bnac.final_metric('ShortAndStrongStimEvokedResponse', sim_row, None, config['main_evoked_comparison_time_window_key'], '', "ALL", config, spontaneous_key=spont_time_window_key)[0])
#   # add_criteria_string(evoked_not_bursting, "All - ShortAndStrongStimEvokedResponse", (evoked_met_x_start, evoked_met_y_start), fpdf)

#   # for i, ng_key in enumerate(config['comparison_ng_keys']):
#   #   evoked_not_bursting = not bool(bnac.final_metric('ShortAndStrongStimEvokedResponse', sim_row, None, config['main_evoked_comparison_time_window_key'], '', ng_key, config, spontaneous_key=spont_time_window_key)[0])
#   #   add_criteria_string(evoked_not_bursting, ng_key + "- ShortAndStrongStimEvokedResponse", (evoked_met_x_indent, evoked_met_y_start + (i+1) * 3), fpdf)


#   # # ######################################################### 2ND EVOKED METRICS

#   # fpdf.set_text_color(0, 0, 0);     
#   # bnac.pdf_txt(fpdf, (2 * raster_width + raster_width/2,subtitle_y), 10, config['secondary_evoked_comparison_time_window_key'])
#   # fpdf.image(config['dirs'][config['secondary_evoked_comparison_time_window_key']]['1stTrialRastersCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRastersCompNGs.png', x=second_evoked_figs_x_start, y=raster_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['secondary_evoked_comparison_time_window_key']]['1stTrialRasterAllGroupsBigNoSpikesALLEI'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_' + '1stTrialRasterAllGroupsBigNoSpikesALLEI.png', x=second_evoked_figs_x_start, y=ei_y, w=raster_width)
#   # fpdf.image(config['dirs'][config['secondary_evoked_comparison_time_window_key']]['MultipleStatsCompNGs'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '_COMPMultipleStatsCompNGs.png', x=second_evoked_figs_x_start + evoked_stats_x_buffer, y=evoked_stats_y, w=evoked_stats_width)




#   # ######################################################### PDF SAVE

#   # sim_summary_pdf_path = sim_row.loc[('simulation', 'paths', 'SimulationFigureOutputDir')] + "Summary.pdf"
#   # fpdf.output(sim_summary_pdf_path,'F')
#   # pages = convert_from_path(sim_summary_pdf_path, 500)
#   # pages[0].save(config['dirs']['InitialAnalysesSummaries'] + sim_row.loc[('simulation', 'paths', 'SimulationString')] + '.png', 'PNG')


#   pdf.save(simulation_row._asdict()['SummaryPNG'])
