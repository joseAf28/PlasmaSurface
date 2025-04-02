import numpy as np
import scipy as sp
import time
import DMsimulator as DMsim
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm


class ErrorPropagation(ABC):
    
    def __init__(self, const_dict, steric_dict, energy_dict_base, file_input_data, max_time=7, print_flag=True):
        
        self.system = DMsim.SurfaceKineticsSimulator(const_dict, steric_dict, file_input_data)
        self.input_data_dict, self.recProbExp_vec = self.system.prepare_data()
        self.const_dict = const_dict
        
        self.energy_dict_base = energy_dict_base.copy()
        self.max_time = max_time
        
        self.print_flag = print_flag
        self.nb_calls = 0
    
    
    def modify_energy_dict(self, counter):
        #### Modify the energy dict based on parameters and counter
        #### This method should be implemented in any class
        pass
    
    
    

    def generate_samples(self, n_samples, mean_input_dict, std_input_dict):
        samples = np.zeros(n_samples, dtype=dict)
        for i in range(n_samples):
            sample = mean_input_dict.copy()
            for key in std_input_dict.keys():
                sample[key] = np.abs(np.random.normal(mean_input_dict[key], std_input_dict[key]))
            
            samples[i] = sample
        return samples
    
    
    def stratified_samples(self, param, n_samples):
        mean, std = param
        quantiles = np.linspace(0, 1, n_samples+2)[1:-1]  # avoid 0 and 1 for stability
        # Add a small random offset within each stratum
        delta = 1/(n_samples+1)
        u_strat = quantiles + (np.random.rand(n_samples)-0.5) * delta
        return sp.stats.norm.ppf(u_strat, loc=mean, scale=std)
    
    
    def generate_stratified_samples(self, n_samples, mean_input_dict, std_input_dict):
        samples = np.zeros(n_samples, dtype=dict)
        for i in range(n_samples):
            sample = mean_input_dict.copy()
            for key in std_input_dict.keys():
                sample[key] = np.abs(self.stratified_samples((mean_input_dict[key], std_input_dict[key]), 1)[0])
            
            samples[i] = sample
        
        samples_array = np.array([list(sample.values()) for sample in samples])
        samples_array_shuffled = samples_array.copy()
        np.random.shuffle(samples_array_shuffled)
        samples = np.array([dict(zip(mean_input_dict.keys(), sample)) for sample in samples_array_shuffled], \
                            dtype=dict)
        
        return samples
    
    
    def sampler_error_propagation(self, std_input_ratios_dict, n_samples):
        
        nb_exp_points = self.input_data_dict.shape[0]
        
        prob_dist_vec = np.zeros((nb_exp_points, n_samples), dtype=float)
        
        pbar = tqdm(total=nb_exp_points * n_samples)
        
        for i in range(nb_exp_points):
            
            mean_input_dict = self.input_data_dict[i]    
            std_input_dict = {key: std_input_ratios_dict[key] * self.input_data_dict[i][key] for key in std_input_ratios_dict.keys()}
        
            samples = self.generate_samples(n_samples, mean_input_dict, std_input_dict)
            
            for j in range(n_samples):
                
                energy_dict_new = self.modify_energy_dict(i)
                
                _, recProb_aux, _, sucess = self.system.solve_system(samples[j], energy_dict_new, solver="fixed_point", max_time=self.max_time)
                
                pbar.update(1)
                
                if sucess == True:
                    prob_dist_vec[i, j] = np.sum(recProb_aux)
                else: 
                    logging.warning(f"Simulation failed for index {i}. Assigning high loss.")
                    prob_dist_vec[i, j] = 0.0
                
            
        pbar.close()
        
        return prob_dist_vec