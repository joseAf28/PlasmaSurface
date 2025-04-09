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
        # u_strat = quantiles + (np.random.rand(n_samples)-0.5) * delta
        u_strat = quantiles + 0.1*(np.random.rand(n_samples)-0.5) * delta
        return sp.stats.norm.ppf(u_strat, loc=mean, scale=std)
    
    
    def generate_stratified_samples(self, n_samples, mean_input_dict, std_input_dict, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for key in mean_input_dict:
                mean = mean_input_dict[key]
                if key in std_input_dict:
                    std = std_input_dict[key]
                    value = np.abs(self.stratified_samples((mean, std), 1)[0])
                else:
                    value = mean
                sample[key] = value
            samples.append(sample)
        
        np.random.shuffle(samples)
        return samples
    
    
    
    def sampler_error_propagation(self, std_input_ratios_dict, n_samples, stratified=False):
        
        ### stratified = False: unbiased estimator
        ### stratified = True: biased estimator, better for the mean estimator
        
        nb_exp_points = self.input_data_dict.shape[0]
        
        prob_dist_vec = np.zeros((nb_exp_points, n_samples), dtype=float)
        
        pbar = tqdm(total=nb_exp_points * n_samples)
        
        for i in range(nb_exp_points):
            
            mean_input_dict = self.input_data_dict[i]    
            std_input_dict = {key: std_input_ratios_dict[key] * self.input_data_dict[i][key] for key in std_input_ratios_dict.keys()}
        
            if stratified:
                samples = self.generate_stratified_samples(n_samples, mean_input_dict, std_input_dict)
            else:
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
    
    
    def compute_mean_func_output(self, func, std_input_ratios_dict, counter):
        ### too slow in the case of many parameters
        
        exp_samples = self.input_data_dict[counter]
        
        variables = len(std_input_ratios_dict)
        mean_input_dict = exp_samples.copy()
        std_input_dict = {key: std_input_ratios_dict[key] * exp_samples[key] for key in std_input_ratios_dict.keys()}
        
        keys_list = list(std_input_ratios_dict.keys())
        
        mean_input_array = np.array([mean_input_dict[key] for key in keys_list])
        std_input_array = np.array([std_input_dict[key] for key in keys_list])
        
        
        def pdf_function(*x):
            prod_func = 1.0
            for i in range(variables):
                prod_func *= sp.stats.norm.pdf(x[i], loc=mean_input_array[i], scale=std_input_array[i])
            
            return prod_func
        
        def integrand(*x):
            
            pdf_value = pdf_function(*x)
            
            sample_mod_dict = {key: x[i] for i, key in enumerate(keys_list)}
            exp_samples.update(sample_mod_dict)
            
            energy_dict_new = self.modify_energy_dict(counter)
            
            _, recProb_aux, _, sucess = self.system.solve_system(exp_samples, energy_dict_new, solver="fixed_point", max_time=self.max_time)
            
            if sucess == True:
                prob_value = np.sum(recProb_aux)
            else: 
                logging.warning(f"Simulation failed for index {counter}. Assigning high loss.")
                prob_value = 0.0
                
            return pdf_value * func(prob_value)
            
        
        bounds = [[mean_input_dict[key] - 4 * std_input_dict[key], mean_input_dict[key] + 4 * std_input_dict[key]] for key in keys_list]
        
        # Perform the integration
        start_time = time.time()
        integral_value, error = sp.integrate.nquad(integrand, bounds)
        end_time = time.time()
        
        logging.info(f"Integration time: {end_time - start_time:.2f} seconds")
        
        return integral_value, error