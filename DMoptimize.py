import numpy as np
import scipy as sp
import time
from abc import ABC, abstractmethod
import DMsimulator as DMsim
import logging


class Optimize(ABC):
    
    def __init__(self, const_dict, steric_dict, energy_dict_base, file_input_data, loss_func, max_time=7, print_flag=True):
        
        self.system = DMsim.SurfaceKineticsSimulator(const_dict, steric_dict, file_input_data)
        self.input_data_dict, self.recProbExp_vec = self.system.prepare_data()
        self.const_dict = const_dict
        
        self.energy_dict_base = energy_dict_base.copy()
        self.loss_func = loss_func
        self.max_time = max_time
        
        self.print_flag = print_flag
    
    
    @abstractmethod
    def modify_energy_dict(self, params, counter):
        #### Modify the energy dict based on parameters and counter
        #### This method should be implemented in any class
        pass
    
    
    def functional_loss(self, params):
        
        loss_vec = np.zeros(len(self.input_data_dict))
        for i in range(len(self.input_data_dict)):
            
            energy_dict_new = self.modify_energy_dict(params, i)
            _, recProb_aux, _, sucess = self.system.solve_system(self.input_data_dict[i], energy_dict_new, solver="fixed_point", max_time=self.max_time)
            
            if sucess == True:
                loss_vec[i] = self.loss_func(np.sum(recProb_aux), self.recProbExp_vec[i])
            else: 
                logging.warning(f"Simulation failed for index {i}. Assigning high loss.")
                loss_vec[i] = 1e6
                
        value = np.sum(loss_vec)
        
        if self.print_flag:
            print("Loss: ", value, "Params: ", params)
        
        return value
    
    
    def hybrid_search(self, config):
        
        bounds = config["bounds"]
        nb_de_calls = config["nb_de_calls"]
        de_maxiter = config["de_maxiter"]
        local_attempts = config["local_attempts"]
        epsilon_local = config["epsilon_local"]
        
        candidates = []
        for _ in range(nb_de_calls):
            result = sp.optimize.differential_evolution(self.functional_loss, bounds, polish=True, disp=True, maxiter=de_maxiter)
            candidates.append(result.x)
            
            if self.print_flag:
                print("Global Search Result: ", result)
                print()
        
        candidates_losses = [self.functional_loss(candidate) for candidate in candidates]
        best_candidate = candidates[np.argmin(candidates_losses)]
        
        if self.print_flag:
            print("Candidates: ", candidates)
            print("Candidates Losses: ", candidates_losses)
            
            print("Best candidate Global Search: ", best_candidate)
            print()
        
        bounds_array = np.array(bounds)
        
        best_local = best_candidate
        best_local_loss = np.min(candidates_losses)
        
        if self.print_flag:
            print("best_local_loss: ", best_local_loss)
            print("best_local: ", best_local)
        
        for attempt in range(local_attempts):
            
            perturbation = np.random.uniform(-epsilon_local, epsilon_local, len(best_candidate))
            x0 = best_local + perturbation

            ### check bounds
            x0 = np.maximum(x0, bounds_array[:, 0]) 
            x0 = np.minimum(x0, bounds_array[:, 1])
            
            # local_result = sp.optimize.minimize(self.functional_loss, x0, method="Nelder-Mead", bounds=bounds)
            local_result = sp.optimize.minimize(self.functional_loss, x0, method="L-BFGS-B", bounds=bounds)
            
            if self.print_flag:
                print("Local Search Attempt: ", attempt, "Loss: ", local_result.fun, "Params: ", local_result.x)
                print()
            
            if local_result.fun < best_local_loss:
                best_local_loss = local_result.fun
                best_local = local_result.x
        
        return best_local, best_local_loss