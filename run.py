from iewIndex.calc_iew import iewIndex

factor_type = ['temp', 'co2']
factor_data = [72, 200]
occ_rate = 1

iew_test = iewIndex()
out = iew_test.calculate_IWQ_index_point(factor_type, factor_data, occ_rate)
print(out)