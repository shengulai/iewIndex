import numpy as np
import json
import plotly.express as px
import pandas as pd

class iewIndex:

    def __init__(self):
        with open('../iewIndex/iewIndex/default_user_weights.json', 'r') as read_file:
            self.all_weights_user = json.loads(read_file.read())
        with open('../iewIndex/iewIndex/default_wellness_range.json', 'r') as read_file:
            self.all_wellness_range = json.loads(read_file.read())


    def calculate_IWQ_index_point(self, factor_type, factor_data, occ_rate):
        """calculate at one timestep, the IWQ index and adjusted weights

        Args:
            factor_type (list): containing all the factor types
                                eg: ['temp', 'co2', 'voc']
            factor_data (list): containing the corresponding factor data to the factor_type
                                eg: [72, 200, 100]
            occ_rate (float): within range [0,1]

        Returns:
            IWQ-index_total: float, within range [0, 1]
            IWQ-index_individual: list
        """
       # check and construct dict with sensor used in the calculation
        self.weights_user = {}
        for i in range(len(factor_type)):
            self.weights_user[factor_type[i]] = self.all_weights_user.get(factor_type[i])
        self.wellness_range = {}
        for i in range(len(factor_type)):
            self.wellness_range[factor_type[i]] = self.all_wellness_range.get(factor_type[i])

        def calculate_err_ratio_2(factor_name, factor_scalar):
            lower_bound = self.wellness_range[factor_name][0]
            upper_bound = self.wellness_range[factor_name][1]

            bad_low = self.wellness_range[factor_name][2]
            bad_up = self.wellness_range[factor_name][3]
            
            # save the equation if two lower or two upper bound are the same
            if lower_bound == bad_low:
                bad_low -= 1
            if upper_bound == bad_up:
                bad_up += 1
            # calculate the err_ratio_2
            if factor_scalar < lower_bound:
                err = lower_bound - factor_scalar
                err_ratio_2 = (err / (lower_bound - bad_low))**2
            elif factor_scalar > upper_bound:
                err = upper_bound - factor_scalar
                err_ratio_2 = (err / (bad_up - upper_bound))**2
            else:
                err = 0
                err_ratio_2 = 0
            return err_ratio_2
            
        def get_weights_user():
            return np.array([self.weights_user[k] for k in factor_type])
        
        def flag_danger_factor(factor_type, factor_data):
            danger_flag = np.zeros(len(factor_type))
            for i in range(len(factor_type)):
                factor_name = factor_type[i]
                factor_scalar = factor_data[i]
                bad_low = self.wellness_range[factor_name][2]
                bad_up = self.wellness_range[factor_name][3]
                if factor_scalar < bad_low or factor_scalar > bad_up:
                    danger_flag[i] = 1
            return danger_flag

        def change_danger_sig_scores(weighted_sig_scores, danger_flag):
            for i in range(np.shape(danger_flag)[0]):
                flag = danger_flag[i]
                if flag == 1:
                    weighted_sig_scores[i] = 1
            return weighted_sig_scores

        self.factor_type = factor_type
        self.factor_data = factor_data
        self.occ_rate = occ_rate
        # calculate error out of comfort limit
        self.err_ratio_2s = []
        for self.factor_name, self.factor_scalar in zip(self.factor_type, self.factor_data):
            self.err_ratio_2s.append(calculate_err_ratio_2(self.factor_name, self.factor_scalar))
        # calculate sigma scores
        self.sig_scores = (np.exp(self.err_ratio_2s)**10 - 1)/(1 + np.exp(self.err_ratio_2s)**10)
        # calculate occupancy weight and weighted sig scores
        self.weight_occ = 1 - np.exp(-10 * self.occ_rate)
        self.weighted_sig_scores = self.weight_occ * get_weights_user() * self.sig_scores
        # factors that is already in danger
        self.danger_flag = flag_danger_factor(self.factor_type, self.factor_data)
        self.weighted_sig_scores = change_danger_sig_scores(self.weighted_sig_scores, self.danger_flag)
        # calculate dynamic weights using softmax function
        self.weights_dynamic = np.exp(self.weighted_sig_scores) / np.sum(np.exp(self.weighted_sig_scores))
        # calculate individual IWQ index 
        self.IWQ_individual = np.round(1 - np.array(self.weighted_sig_scores),2)
        # calcuate the total IWQ with dynamic weights
        self.IWQ_total = np.round(np.sum(self.IWQ_individual * self.weights_dynamic),2)


        ## Outputs
        self.IWQ_out = {}
        self.IWQ_out['IEW_Index'] = int(self.IWQ_total*100)

        self.factorIndex = {}
        self.factorWeight = {}
        for i in range(len(factor_type)):
            self.factorIndex[factor_type[i]] = int(self.IWQ_individual[i]*100)
            self.factorWeight[factor_type[i]] = self.weights_dynamic[i]

        self.IWQ_out['factorIndex'] = self.factorIndex
        self.IWQ_out['factorWeight'] = self.factorWeight

        return self.IWQ_out

        # return self.IWQ_total*100, self.IWQ_individual*100, self.weights_dynamic


    def build_graph_df(self, df_data_list, time_of_interest, occ_rate, zones):
            """
            To build the dataFrame for the sunburst chart
            
            Returns:
                df_sunburst (pandas.DataFrame): the dataFrame for sunburst chart
            
            """
            self.df_sunburst = pd.DataFrame(columns=['zone', 'factor_type', 'factor_weight', 'factor_iwq',
                                    'factor_mean_iwq', 'zone_mean_iwq', 'building_iwq'])
            for i in range(len(df_data_list)):
                df_data = df_data_list[i]
                factor_types = list(df_data.columns)
                df_temp = pd.DataFrame(columns=['zone', 'factor_type', 'factor_data', 
                                                'factor_weight', 'factor_iwq', 'zone_mean_iwq'])
                df_temp['factor_type'] = factor_types
                df_temp['factor_weight'] = [self.all_weights_user[k] for k in factor_types]
                df_temp['zone'] = zones[i]
                factor_data = list(df_data.loc[time_of_interest].values)
                df_temp['factor_data'] = factor_data
                IWQ_total, IWQ_individual, weights_dynamic = self.calculate_IWQ_index_point(factor_types, factor_data, occ_rate)
                df_temp['zone_mean_iwq'] = round(IWQ_total,3)
                df_temp['factor_iwq'] = [round(k,3) for k in IWQ_individual]
                self.df_sunburst = self.df_sunburst.append(df_temp, ignore_index=True)
            self.df_sunburst['factor_mean_iwq'] = round(self.df_sunburst.groupby('factor_type')['factor_iwq'].transform('mean'),3)
            self.df_sunburst['zone_mean_iwq'] = round(self.df_sunburst.groupby('zone')['factor_iwq'].transform('mean'),3)
            self.df_sunburst['building_iwq'] = round(self.df_sunburst['factor_iwq'].mean(), 3)
            return self.df_sunburst


    def show_donut(self, time_of_interest, occ_rate, df_data_list, zones, range_color=None, factor_first=True):
        """To visualize the IWQ-index in a sunburst chart style

        Args:
            time_of_interest (pandas.datetime): the time point to visualize
            occ_rate (float): within range [0,1]
            df_data_list (a list of pandas.DataFrame): MODIFY HERE if the zones are not in separated dataFrames
            zones (a list of strings): The zone name list
                                eg: ['Floor1', 'Floor2']
            range_color (a list with two elements): To specify the color range
            factor_first (Boolean): To choose the sequence of the factor and the zone
        """
        self.df_sunburst = self.build_graph_df(df_data_list, time_of_interest, occ_rate, zones)
        if factor_first:
            path=['building_iwq', 'factor_type', 'zone']
        else:
            path=['building_iwq', 'zone', 'factor_type']
        fig = px.sunburst(self.df_sunburst, path=path, values='factor_weight',
                      color='factor_iwq', hover_data=['factor_mean_iwq', 'zone_mean_iwq'], 
                      range_color=range_color, color_continuous_scale='RdBu')
        fig.show()