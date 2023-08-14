import pandas as pd
from blueetl.utils import load_yaml
from blueetl.analysis import Analyzer
from barreletl.utils import merge_all_features
import sys

def ji_merge_features(parameters, a, feat):

	# import pdb; pdb.set_trace()

	features_to_concat = []
	for feature_str in list(a.features._data.keys()):
		if (feat in feature_str):

			prefix = feature_str.split(feat)[0] + "_"
			# print(prefix)

			# import pdb; pdb.set_trace()

			# print(prefix)
			
			a.features._data[feature_str].df.loc[:, "bin_size"] = parameters.etl.q(prefix=prefix).bin_size.values[0]
			a.features._data[feature_str].df.loc[:, "sigma"] = parameters.etl.q(prefix=prefix).sigma.values[0]

			features_to_concat.append(a.features._data[feature_str].df)

	return pd.concat(features_to_concat)

def merge_features(a):

	
	print("Merge features")
	

	features_path = a.analysis_config["output"].joinpath("features")
	# print(features_path)
	analysis_params = []
	for setup in a.analysis_config["analysis"]["features"]:
		for param_key in setup["params"].keys():
			if "baseline_PSTH" in param_key:
				prefix = ""
				if "prefix" in setup["params"]:
					prefix = setup["params"]["prefix"]

				params = setup["params"][f"{prefix}baseline_PSTH"]["params"]
				params["prefix"] = prefix

				analysis_params.append(params)
	analysis_params = pd.DataFrame(analysis_params)
	a.features._data['analysis_params'] = analysis_params

	# print(analysis_params)
	# import pdb; pdb.set_trace()
	# print("Merge features files")
	baseline = ji_merge_features(analysis_params, a, "_baseline_PSTH")
	decay = ji_merge_features(analysis_params, a, "_decay")
	latency = ji_merge_features(analysis_params, a, "_latency")


	# baseline = ji_merge_features(
	#     analysis_params, str(features_path) + "/*_baseline_PSTH.parquet", -21
	# )
	# # baseline.reset_index().to_parquet(
	# #     str(a.analysis_config["output"]) + "/ALL_baseline_PSTH.parquet", engine="pyarrow"
	# # )

	# decay = ji_merge_features(
	#     analysis_params, str(features_path) + "/*_decay.parquet", -13
	# )
	# # decay.reset_index().to_parquet(
	# #     str(a.analysis_config["output"]) + "/ALL_decay.parquet", engine="pyarrow"
	# # )

	# latency = ji_merge_features(
	#     analysis_params, str(features_path) + "/*latency.parquet", -15
	# )
	# # latency.reset_index().to_parquet(
	# #     str(a.analysis_config["output"]) + "/ALL_latency.parquet", engine="pyarrow"
	# # )


	a.features._data['baseline'] = baseline.reset_index()
	a.features._data['decay'] = decay.reset_index()
	a.features._data['latency'] = latency.reset_index()

	# print("FEATURES MERGES")
	# sys.exit()