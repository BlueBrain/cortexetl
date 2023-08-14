import os
import glob

proj83_output_root = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/blueetl_ji_output/"
cortex_etl_output_root = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortex_etl_output/"
manuscript_figures_dir = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/manuscript_figures/"

cond_campaign_dict = {
	
	"unconnected": {"src": proj83_output_root + "ConductanceCallibrationSSCxO1-Campaign1/ConductanceCallibrationSSCxO1-Campaign1-UnconnectedScan/figures/",
					"dst": "0-Unconnected/"},

	"initial_connection": {"src": cortex_etl_output_root + "1-InitialCombination/1-InitialCombination-1-1stConnection/hex0_spikes/figures/",
					"dst": "1-InitialConnection/"},

	"5th_connection": {"src": cortex_etl_output_root + "1-InitialCombination/1-InitialCombination-5-5thConnection/hex0_spikes/figures/",
					"dst": "2-5thConnection/"},

	"transfer_3rd_iter_single_example": {"src": cortex_etl_output_root + "2-PfrTransfer/2-PfrTransfer-3-3rdConnection/hex0_spikes/figures/",
					"dst": "3-Transfer3rdIterationSingleExample/"},

	"transfer_final_all_combinations": {"src": cortex_etl_output_root + "2-PfrTransfer/2-PfrTransfer-6-3rdConnectionRemaining/hex0_spikes/figures/",
					"dst": "4-TransferFinalAllCombinations/"},

	"single_whisker_deflection_O1": {"src": cortex_etl_output_root + "3-ThalamicStimuli/3-ThalamicStimuli-MegaScan2Scalings/hex0_spikes/1.36/figures/",
					"dst": "7-SingleWhiskerDeflection-O1/"},

	"active_touch_O1": {"src": cortex_etl_output_root + "7-ActiveWhiskerTouch/7-ActiveWhiskerTouch-Test0/hex0_spikes/figures/",
					"dst": "8-ActiveTouch-O1/"},

	"spontaneous_activity_FullSSCx_hex0": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-FprScan/hex0_spikes/figures/",
					"dst": "10-SpontaneousActivity-FullSSCx-hex0/"},

	"spontaneous_activity_FullSSCx_hex59": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-FprScan/hex59_spikes/figures/",
					"dst": "10-SpontaneousActivity-FullSSCx-hex59/"},

	"spontaneous_activity_FullSSCx_hexes": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-FprScan/hexes_spikes/figures/",
					"dst": "10-SpontaneousActivity-FullSSCx-hexes/"},

	"spontaneous_activity_FullSSCx_all": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-FprScan/AllCompartments_spikes/figures/",
					"dst": "10-SpontaneousActivity-FullSSCx-all/"},

	"single_whisker_deflection_FullSSCx_hex0": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-Fpr15-StimScan-10x/hex0_spikes/figures/",
					"dst": "11-SingleWhiskerDeflection-FullSSCx_hex0/"},

	"single_whisker_deflection_FullSSCx_all": {"src": cortex_etl_output_root + "5-FullCircuit/5-FullCircuit-2-BetterMinis-Fpr15-StimScan-10x/AllCompartments_spikes/figures/",
					"dst": "11-SingleWhiskerDeflection-FullSSCx_all/"},

}



def copy_f(original_fig_path, new_fig_path, override_check=False):

	# print("cp " + original_fig_path)
	if ((override_check) or os.path.exists(original_fig_path)):
		# shutil.copy(original_fig_path, new_fig_path)
		os.system("cp " + original_fig_path + " " + new_fig_path)
	else:
		print(original_fig_path + " DOESN'T EXIST")




def copy_manuscript_figures(campaigns_dict, depol_type, manuscript_figures_root):

	depol_manuscript_fig_root = manuscript_figures_root + '/'
	os.makedirs(depol_manuscript_fig_root, exist_ok=True)

	# if (create_videos):
		# sw.create_gif_from_list_of_image_file_names(glob.glob(orig_state_anal_figs_root + "Conj/PopulationState/StatesByTrial/Plots/*_Conj_StatesByTrialAdjusted.png"), popstate_state_analysis_fig_dir + 'PopStateExamplesOriginal.mp4')


	for campaign_key in list(campaigns_dict.keys()):

		campaign_dict = campaigns_dict[campaign_key]
		src = campaign_dict['src']
		dst = depol_manuscript_fig_root + campaign_dict['dst']
		os.makedirs(dst, exist_ok=True)

		if (campaign_key == "unconnected"):

			copy_f(src + "UnconnectedFRGrid_ALL.pdf", dst + "UNCONNECTED_SCAN_FRGrid.pdf")
			copy_f(src + "UnconnectedFRGrid_L5_EXC.pdf", dst + "UNCONNECTED_SCAN_FR_L5EXC.pdf")


		###### INITIAL SPONT SCAN ######
		if (campaign_key == "initial_connection"):

			copy_f(src + "SingProportionOfInVivo_NonBursting.pdf", 																	dst + "1stConn-A-ProportionsNonBursting.pdf")
			copy_f(src + "FRs_NonBursting.pdf", 																					dst + "1stConn-B-FRsNonBursting.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "1stConn-C-OUMeans.pdf")
			copy_f(src + "fr_condition_comparisons/nc_frs_desired_unconnected_fr_VS_unconn_2nd_half_.pdf", 							dst + "1stConn-D-UnconnectedValidation.pdf")
			copy_f(src + "fr_condition_comparisons/nc_frs_desired_connected_fr_VS_conn_spont_bursting.pdf", 						dst + "5thConn-E-NGDesConnVsConn.pdf")
			# copy_f(src + "fr_condition_comparisons/nc_frs_desired_unconnected_fr_VS_conn_spont_bursting_or_fr_above_threshold.pdf", dst + "5thConn-F-NGDesUnconnVsConn.pdf")


		if (campaign_key == "5th_connection"):

			copy_f(src + "SingProportionOfInVivo_NonBursting.pdf", 																	dst + "5thConn-A-ProportionsNonBursting.pdf")
			copy_f(src + "FRs_NonBursting.pdf", 																					dst + "5thConn-B-NonBursting.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "5thConn-C-OUMeans.pdf")
			copy_f(src + "campaign_comparison/LP_euc_dist_to_desired_proportion_of_in_vivo_FRs_bursting.pdf", 						dst + "5thConn-D-ScaledEucDist5Camps.pdf")
			copy_f(src + "DesiredUnconnectedFRs.pdf", 																				dst + "5thConn-E-DesiredUnconnectedFRs.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "5thConn-F-DepolMNonBursting.pdf")
			copy_f(src + "fr_condition_comparisons/nc_frs_desired_unconnected_fr_VS_unconn_2nd_half_.pdf", 							dst + "5thConn-G-UnconnectedValidation.pdf")
			copy_f(src + "fr_condition_comparisons/nc_frs_desired_connected_fr_VS_conn_spont_bursting.pdf", 						dst + "5thConn-H-NGDesConnVsConn.pdf")
			# copy_f(src + "fr_condition_comparisons/nc_frs_desired_unconnected_fr_VS_conn_spont_bursting_or_fr_above_threshold.pdf", dst + "5thConn-I-NGDesUnconnVsConn.pdf")
			

			copy_f(src + "campaign_comparison/SingProportionOfInVivo_NonBursting.mp4", 												dst + "5thConn-V1-SingProportionsNonBursting.mp4")
			copy_f(src + "campaign_comparison/FRs_NonBursting.mp4", 																dst + "5thConn-V2-NonBursting.mp4")
			copy_f(src + "campaign_comparison/DepolM_NonBursting.mp4", 																dst + "5thConn-V3-OUMeans.mp4")
			copy_f(src + "campaign_comparison/DesiredUnconnectedFRs.mp4", 															dst + "5thConn-V4-DesiredUnconnectedFRs.mp4")

		if (campaign_key == "transfer_3rd_iter_single_example"):

			copy_f(src + "SingProportionOfInVivo_NonBursting.pdf", 																	dst + "Transfer3rdConn-A-ProportionsNonBursting.pdf")
			copy_f(src + "FRs_NonBursting.pdf", 																					dst + "Transfer3rdConn-B-NonBursting.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "Transfer3rdConn-C-OUMeans.pdf")
			copy_f(src + "campaign_comparison/LP_euc_dist_to_desired_proportion_of_in_vivo_FRs_bursting.pdf", 						dst + "Transfer3rdConn-D-ScaledEucDist5Camps.pdf")
			copy_f(src + "DesiredUnconnectedFRs.pdf", 																				dst + "Transfer3rdConn-E-DesiredUnconnectedFRs.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "Transfer3rdConn-F-DepolMNonBursting.pdf")
			
			copy_f(src + "campaign_comparison/SingProportionOfInVivo_NonBursting.mp4", 												dst + "Transfer3rdConn-V1-SingProportionsNonBursting.mp4")
			copy_f(src + "campaign_comparison/FRs_NonBursting.mp4", 																dst + "Transfer3rdConn-V2-NonBursting.mp4")
			copy_f(src + "campaign_comparison/DepolM_NonBursting.mp4", 																dst + "Transfer3rdConn-V3-OUMeans.mp4")
			copy_f(src + "campaign_comparison/DesiredUnconnectedFRs.mp4", 															dst + "Transfer3rdConn-V4-DesiredUnconnectedFRs.mp4")

			# Might still copy: 
				# fr_condition_comparisons

		if (campaign_key == "transfer_final_all_combinations"):

			copy_f(src + "FRs_NonBursting.pdf", 																					dst + "TransferFinalAllCombs-A-FRs.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "TransferFinalAllCombs-B-DepolM.pdf")
			copy_f(src + "MeanConductanceInjection_NonBursting.pdf", 																dst + "TransferFinalAllCombs-C-MeanConductanceInjection.pdf")
			copy_f(src + "ConnUnconnProp_NonBursting.pdf", 																			dst + "TransferFinalAllCombs-D-ConnUnconnPropj.pdf")
			copy_f(src + "DesiredUnconnectedFRs.pdf", 																				dst + "TransferFinalAllCombs-E-DesiredUnconnectedFRs.pdf")
			copy_f(src + "LayerWiseEI_NonBursting.pdf", 																			dst + "TransferFinalAllCombs-F-LayerWiseEI_NonBursting.pdf")
			copy_f(src + "missing_E_synapses_VS_true_mean_conductance.pdf", 														dst + "TransferFinalAllCombs-G-missing_E_synapses_VS_true_mean_conductance.pdf")
			copy_f(src + "missing_E_synapses_VS_resting_conductance.pdf", 															dst + "TransferFinalAllCombs-H-missing_E_synapses_VS_resting_conductance.pdf")
			copy_f(src + "missing_E_synapses_VS_depol_mean.pdf", 																	dst + "TransferFinalAllCombs-I-missing_E_synapses_VS_depol_mean.pdf")
			copy_f(src + "BurstingRatio.pdf", 																						dst + "TransferFinalAllCombs-J-BurstingRatio.pdf")
			copy_f(src + "SingProportionOfInVivo_NonBursting.pdf", 																	dst + "TransferFinalAllCombs-K-SingProportionOfInVivo_NonBursting.pdf")
			copy_f(src + "ProportionOfInVivo_NonBursting.pdf", 																		dst + "TransferFinalAllCombs-L-ProportionOfInVivo_NonBursting.pdf")

			copy_f(src + "heatmaps/FFT_M-bursting.pdf", 																			dst + "TransferFinalAllCombs-M-FFT_M-bursting.pdf")
			copy_f(src + "heatmaps/ei_corr_rval.pdf", 																				dst + "TransferFinalAllCombs-N-heatmaps.pdf")
			copy_f(src + "ProportionOfInVivo_NonBursting.pdf", 																		dst + "TransferFinalAllCombs-O-ProportionOfInVivo_NonBursting.pdf")



			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_:False.mp4", 					dst + "TransferFinalAllCombs-P-RasterVid1.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_S_Gaussian_3.0_1.0_YE__:False.mp4", 							dst + "TransferFinalAllCombs-Q-RasterVid2.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_:False.mp4", 						dst + "TransferFinalAllCombs-R-RasterVid3.mp4")


			copy_f(src + "individual_simulations/\(1.05\,\ 0.9\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_RASTER.pdf", 	dst + "TransferFinalAllCombs-S-RasterExample-A1.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.9\,\ 0.4\)/Rasters/conn_spont_S_Gaussian_3.0_1.0_YE__RASTER.pdf", 				dst + "TransferFinalAllCombs-S-RasterExample-A2.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.9\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_RASTER.pdf", 		dst + "TransferFinalAllCombs-S-RasterExample-A3.pdf", override_check=True)

			copy_f(src + "individual_simulations/\(1.1\,\ 0.9\,\ 0.2\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_RASTER.pdf", 		dst + "TransferFinalAllCombs-S-RasterExample-B1.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.1\,\ 0.9\,\ 0.2\)/Rasters/conn_spont_S_Gaussian_3.0_1.0_YE__RASTER.pdf", 				dst + "TransferFinalAllCombs-S-RasterExample-B2.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.1\,\ 0.9\,\ 0.2\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_RASTER.pdf", 			dst + "TransferFinalAllCombs-S-RasterExample-B3.pdf", override_check=True)

			fr_test_path = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/blueetl_ji_output/sscx_calibration_mgfix/8-FRDistributionTest/8-FRDistributionTest-0-Test/hex0/figures/"
			copy_f(fr_test_path + "MFR-Violin.pdf", 																				dst + "TransferFinalAllCombs-T-FRDist1.pdf")
			copy_f(fr_test_path + "MFR-SpikingOnly-Violin.pdf", 																	dst + "TransferFinalAllCombs-T-FRDist2.pdf")

			flatspace_vid_path = src + "../../hex_O1_spikes/figures/individual_simulations/\(1.1\,\ 0.9\,\ 0.2\)/FlatspaceVideos/conn_spont_100.0_1500_1.0_38/"
			copy_f(flatspace_vid_path + "frame0011.png", dst + "TransferFinalAllCombs-U-SpatialExampleF0.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0012.png", dst + "TransferFinalAllCombs-U-SpatialExampleF1.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0013.png", dst + "TransferFinalAllCombs-U-SpatialExampleF2.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0014.png", dst + "TransferFinalAllCombs-U-SpatialExampleF3.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0015.png", dst + "TransferFinalAllCombs-U-SpatialExampleF4.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0016.png", dst + "TransferFinalAllCombs-U-SpatialExampleF5.png", override_check=True)
			copy_f(flatspace_vid_path + "frame0017.png", dst + "TransferFinalAllCombs-U-SpatialExampleF6.png", override_check=True)

			copy_f(src + "lineplots/LP_euc_dist_to_scaled_in_vivo_FRs_bursting.pdf", 												dst + "TransferFinalAllCombs-V-euc_dist_to_scaled_in_vivo_FRs.pdf")
			copy_f(src + "lineplots/LP_euc_dist_to_desired_proportion_of_in_vivo_FRs_bursting.pdf", 								dst + "TransferFinalAllCombs-V-euc_dist_to_desired_proportion_of_in_vivo_FRs.pdf")


			# Might still copy: 
				# fr_condition_comparisons
				# campaign_comparison

		


		if (campaign_key == "single_whisker_deflection_O1"):

			copy_f(src + "../../no_filter/figures/evoked/L5E_scaling_ratio_comparison.pdf", 								dst + "L5E_scaling_ratio_comparison.pdf")

			copy_f(src + "evoked/none_EvokedRatios.pdf", 																	dst + "none_EvokedRatios.pdf")
			copy_f(src + "evoked/mean_normalise_EvokedRatios.pdf", 															dst + "mean_normalise_EvokedRatios.pdf")
			copy_f(src + "evoked/heatmaps/mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf", 				dst + "mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf")

			copy_f(src + "evoked/time_course_comp/ReyesPuertaPVSST_latencies.pdf", 											dst + "ReyesPuertaPVSST_latencies.pdf")
			copy_f(src + "evoked/time_course_comp/ReyesPuertaEI_latencies.pdf", 											dst + "ReyesPuertaEI_latencies.pdf")

			copy_f(src + "evoked/psths/short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf",										dst + "short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf")
			copy_f(src + "evoked/psths/long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf",											dst + "long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf")


		if (campaign_key == "active_touch_O1"):

			copy_f(src + "evoked/none_EvokedRatios.pdf", 																	dst + "none_EvokedRatios.pdf")
			copy_f(src + "evoked/mean_normalise_EvokedRatios.pdf", 															dst + "mean_normalise_EvokedRatios.pdf")
			copy_f(src + "evoked/heatmaps/mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf", 				dst + "mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf")

			copy_f(src + "evoked/time_course_comp/ReyesPuertaPVSST_latencies.pdf", 											dst + "ReyesPuertaPVSST_latencies.pdf")
			copy_f(src + "evoked/time_course_comp/ReyesPuertaEI_latencies.pdf", 											dst + "ReyesPuertaEI_latencies.pdf")

			copy_f(src + "evoked/psths/short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf",										dst + "short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf")
			copy_f(src + "evoked/psths/long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf",											dst + "long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf")












		#######################
		### Full SSCx spont ###


		if (campaign_key == "spontaneous_activity_FullSSCx_hex0"):

			copy_f(src + "FRs_NonBursting.pdf", 																					dst + "TransferFinalAllCombs-A-FRs.pdf")
			copy_f(src + "DepolM_NonBursting.pdf", 																					dst + "TransferFinalAllCombs-B-DepolM.pdf")
			copy_f(src + "MeanConductanceInjection_NonBursting.pdf", 																dst + "TransferFinalAllCombs-C-MeanConductanceInjection.pdf")
			copy_f(src + "ConnUnconnProp_NonBursting.pdf", 																			dst + "TransferFinalAllCombs-D-ConnUnconnPropj.pdf")
			copy_f(src + "DesiredUnconnectedFRs.pdf", 																				dst + "TransferFinalAllCombs-E-DesiredUnconnectedFRs.pdf")
			copy_f(src + "LayerWiseEI_NonBursting.pdf", 																			dst + "TransferFinalAllCombs-F-LayerWiseEI_NonBursting.pdf")
			copy_f(src + "BurstingRatio.pdf", 																						dst + "TransferFinalAllCombs-J-BurstingRatio.pdf")
			copy_f(src + "SingProportionOfInVivo_NonBursting.pdf", 																	dst + "TransferFinalAllCombs-K-SingProportionOfInVivo_NonBursting.pdf")
			copy_f(src + "ProportionOfInVivo_NonBursting.pdf", 																		dst + "TransferFinalAllCombs-L-ProportionOfInVivo_NonBursting.pdf")

			copy_f(src + "heatmaps/FFT_M-bursting.pdf", 																			dst + "TransferFinalAllCombs-M-FFT_M-bursting.pdf")
			copy_f(src + "heatmaps/ei_corr_rval.pdf", 																				dst + "TransferFinalAllCombs-N-heatmaps.pdf")
			copy_f(src + "ProportionOfInVivo_NonBursting.pdf", 																		dst + "TransferFinalAllCombs-O-ProportionOfInVivo_NonBursting.pdf")

			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_:False.mp4", 					dst + "TransferFinalAllCombs-P-RasterVid1.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_S_Gaussian_3.0_1.0_YE__:False.mp4", 							dst + "TransferFinalAllCombs-Q-RasterVid2.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_:False.mp4", 						dst + "TransferFinalAllCombs-R-RasterVid3.mp4")

			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_RASTER.pdf", 	dst + "TransferFinalAllCombs-S-RasterExample-A1.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_S_Gaussian_3.0_1.0_YE__RASTER.pdf", 				dst + "TransferFinalAllCombs-S-RasterExample-A2.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_RASTER.pdf", 		dst + "TransferFinalAllCombs-S-RasterExample-A3.pdf", override_check=True)

			# fr_test_path = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/blueetl_ji_output/sscx_calibration_mgfix/8-FRDistributionTest/8-FRDistributionTest-0-Test/hex0/figures/"
			# copy_f(fr_test_path + "MFR-Violin.pdf", 																				dst + "TransferFinalAllCombs-T-FRDist1.pdf")
			# copy_f(fr_test_path + "MFR-SpikingOnly-Violin.pdf", 																	dst + "TransferFinalAllCombs-T-FRDist2.pdf")

			copy_f(src + "lineplots/LP_euc_dist_to_scaled_in_vivo_FRs_bursting.pdf", 												dst + "TransferFinalAllCombs-V-euc_dist_to_scaled_in_vivo_FRs.pdf")
			copy_f(src + "lineplots/LP_euc_dist_to_desired_proportion_of_in_vivo_FRs_bursting.pdf", 								dst + "TransferFinalAllCombs-V-euc_dist_to_desired_proportion_of_in_vivo_FRs.pdf")

		
		if (campaign_key == "spontaneous_activity_FullSSCx_hex59"):

			copy_f(src + "FRs_.pdf", 																								dst + "TransferFinalAllCombs-A-FRs.pdf")
			copy_f(src + "DepolM_.pdf", 																							dst + "TransferFinalAllCombs-B-DepolM.pdf")
			copy_f(src + "MeanConductanceInjection_.pdf", 																			dst + "TransferFinalAllCombs-C-MeanConductanceInjection.pdf")
			copy_f(src + "ConnUnconnProp_.pdf", 																					dst + "TransferFinalAllCombs-D-ConnUnconnPropj.pdf")
			copy_f(src + "DesiredUnconnectedFRs.pdf", 																				dst + "TransferFinalAllCombs-E-DesiredUnconnectedFRs.pdf")
			copy_f(src + "LayerWiseEI_.pdf", 																						dst + "TransferFinalAllCombs-F-LayerWiseEI.pdf")
			copy_f(src + "BurstingRatio.pdf", 																						dst + "TransferFinalAllCombs-J-BurstingRatio.pdf")
			copy_f(src + "SingProportionOfInVivo_.pdf", 																			dst + "TransferFinalAllCombs-K-SingProportionOfInVivo.pdf")
			copy_f(src + "ProportionOfInVivo_.pdf", 																				dst + "TransferFinalAllCombs-L-ProportionOfInVivo.pdf")

			copy_f(src + "heatmaps/FFT.pdf", 																						dst + "TransferFinalAllCombs-M-FFT_M-bursting.pdf")
			copy_f(src + "heatmaps/ei_corr_rval.pdf", 																				dst + "TransferFinalAllCombs-N-heatmaps.pdf")
			copy_f(src + "ProportionOfInVivo_.pdf", 																				dst + "TransferFinalAllCombs-O-ProportionOfInVivo.pdf")

			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_:False.mp4", 					dst + "TransferFinalAllCombs-P-RasterVid1.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_S_Gaussian_3.0_1.0_YE__:False.mp4", 							dst + "TransferFinalAllCombs-Q-RasterVid2.mp4")
			copy_f(src + "raster_videos/conn_spont/False/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_:False.mp4", 						dst + "TransferFinalAllCombs-R-RasterVid3.mp4")

			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_LayerEI_RASTER.pdf", 	dst + "TransferFinalAllCombs-S-RasterExample-A1.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_S_Gaussian_3.0_1.0_YE__RASTER.pdf", 				dst + "TransferFinalAllCombs-S-RasterExample-A2.pdf", override_check=True)
			copy_f(src + "individual_simulations/\(1.05\,\ 0.15\,\ 0.4\)/Rasters/conn_spont_NS_Gaussian_3.0_1.0_YNE_All_RASTER.pdf", 		dst + "TransferFinalAllCombs-S-RasterExample-A3.pdf", override_check=True)

			# fr_test_path = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/blueetl_ji_output/sscx_calibration_mgfix/8-FRDistributionTest/8-FRDistributionTest-0-Test/hex0/figures/"
			# copy_f(fr_test_path + "MFR-Violin.pdf", 																				dst + "TransferFinalAllCombs-T-FRDist1.pdf")
			# copy_f(fr_test_path + "MFR-SpikingOnly-Violin.pdf", 																	dst + "TransferFinalAllCombs-T-FRDist2.pdf")


			copy_f(src + "lineplots/LP_euc_dist_to_scaled_in_vivo_FRs_bursting.pdf", 												dst + "TransferFinalAllCombs-V-euc_dist_to_scaled_in_vivo_FRs.pdf")
			copy_f(src + "lineplots/LP_euc_dist_to_desired_proportion_of_in_vivo_FRs_bursting.pdf", 								dst + "TransferFinalAllCombs-V-euc_dist_to_desired_proportion_of_in_vivo_FRs.pdf")


		if (campaign_key == "spontaneous_activity_FullSSCx_hexes"):

			copy_f(src + "multi_hex/rvals_by_hex_2.pdf", 																			dst + "MultiHex_2_RVal.pdf")
			copy_f(src + "multi_hex/rvals_by_hex_3.pdf", 																			dst + "MultiHex_3_RVal.pdf")
			copy_f(src + "multi_hex/rvals_by_hex_4.pdf", 																			dst + "MultiHex_4_RVal.pdf")
			copy_f(src + "multi_hex/rvals_by_hex_5.pdf", 																			dst + "MultiHex_5_RVal.pdf")
			copy_f(src + "multi_hex/rvals_by_hex_6.pdf", 																			dst + "MultiHex_6_RVal.pdf")


		if (campaign_key == "spontaneous_activity_FullSSCx_all"):

			# flatspace_vid_path = src + "flatspace_videos/conn_spont_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\).mp4"
			copy_f(src + "flatspace_videos/conn_spont_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\).mp4", 						dst + "conn_spont_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\).mp4", override_check=True)
			copy_f(src + "flatspace_videos/conn_spont_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", 				dst + "conn_spont_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", override_check=True)



		#######################
		### Full SSCx evok ###

		if (campaign_key == "single_whisker_deflection_FullSSCx_hex0"):

			copy_f(src + "evoked/none_EvokedRatios.pdf", 																	dst + "none_EvokedRatios.pdf")
			copy_f(src + "evoked/mean_normalise_EvokedRatios.pdf", 															dst + "mean_normalise_EvokedRatios.pdf")
			copy_f(src + "evoked/heatmaps/mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf", 				dst + "mean_ratio_difference_M-SimBad100p50p25pDecayOverlySustained.pdf")

			copy_f(src + "evoked/time_course_comp/ReyesPuertaPVSST_latencies.pdf", 											dst + "ReyesPuertaPVSST_latencies.pdf")
			copy_f(src + "evoked/time_course_comp/ReyesPuertaEI_latencies.pdf", 											dst + "ReyesPuertaEI_latencies.pdf")

			copy_f(src + "evoked/psths/short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf",											dst + "short_without_sim_bad_1.0_1GROUPED_PSTHs.pdf")
			copy_f(src + "evoked/psths/long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf",											dst + "long_with_sim_bad_1.0_3GROUPED_PSTHs.pdf")



		if (campaign_key == "single_whisker_deflection_FullSSCx_all"):

			# flatspace_vid_path = src + "../../hex_O1_spikes/figures/individual_simulations/\(1.1\,\ 0.9\,\ 0.2\)/FlatspaceVideos/conn_spont_100.0_1500_1.0_38/"
			# copy_f(flatspace_vid_path + "frame0011.png", dst + "TransferFinalAllCombs-U-SpatialExampleF0.png", override_check=True)

			copy_f(src + "flatspace_videos/evoked_SOZ_500ms_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\).mp4", 							dst + "evoked_SOZ_500ms_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\).mp4", override_check=True)
			copy_f(src + "flatspace_videos/evoked_SOZ_500ms_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", 				dst + "evoked_SOZ_500ms_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", override_check=True)

			copy_f(src + "flatspace_videos/full_sim_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\).mp4", 							dst + "full_sim_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\).mp4", override_check=True)
			copy_f(src + "flatspace_videos/full_sim_50.0_20000_1.0/3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", 				dst + "full_sim_50.0_20000_1.0_3_\(1.05\,\ 0.15\,\ 0.4\)_hist_mean.pdf", override_check=True)



		



copy_manuscript_figures(cond_campaign_dict, "COND", manuscript_figures_dir)
