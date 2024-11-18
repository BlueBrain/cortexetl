import sys
sys.path.append('.')
import cortexetl as c_etl

def main():

	mas = []

	import joblib
	print(joblib.cpu_count())

	# print(sys.argv[1:])
	for analysis_config_path in sys.argv[1:]:

		ma = c_etl.analysis_initial_processing(analysis_config_path, loglevel='ERROR')
		for a_name, a in ma.analyzers.items():
			c_etl.apply_analyses(a, a_name=a_name)
		mas.append(ma)

	return mas


if __name__ == "__main__":
	mas = main()