import numpy as np
import scipy as sp
import h5py
import time


class SurfaceKineticsSimulator():
    
    def __init__(self, const_dict, SF_dict, file_data="Experimental_data_TD.hdf5", init_conditions=[0.1, 0.1, 1.0, 0.0, 0.0], timeSpace=np.linspace(0, 1_00, 5_00)):
        self.const_dict = const_dict
        self.init_conditions = init_conditions
        self.timeSpace = timeSpace
        self.file_exp_data = file_data
        self.SF_dict = SF_dict
    
    
    def prepare_data(self):
        #### the .hdf5 file was obtained by merging the data from the .xlsx files
        
        ### Read the experimental data from the hdf5 file
        data_dict = {}
        with h5py.File(self.file_exp_data, "r") as file:
            keys = list(file.keys())
            
            for key in keys:
                data_dict[key] = file[key][:]
        
        file.close()
        

        ####* Prepara the data for the simulation
        Tnw_vec = data_dict["TnwExp"]
        Tw_vec = data_dict["Twall"]
        
        Oden_vec = data_dict["OmeanExp"]
        Pressure_vec = data_dict["Pressure"]
        Recomb_prob_exp_vec = data_dict["recProbExp"]
        FluxIon_vec = 1e14 * data_dict["Current"]
        
        
        try: 
            N_vec = data_dict["NExp"]
        except:
            N_vec = np.zeros_like(Tnw_vec)
        
        #### EavgMB data
        p_data_exp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.5]
        EavgMB_data = [1.04, 0.91, 0.87, 0.83, 0.77, 0.5, 0.001]
        interpolator = sp.interpolate.interp1d(p_data_exp, EavgMB_data, kind='linear', fill_value=0.001, bounds_error=False)
        
        EavgMB_vec = np.array(interpolator(p_data_exp))
        
        exp_vec = []
        for i in range(len(Tnw_vec)):
            exp_vec.append({
                "Tnw": Tnw_vec[i], "Tw": Tw_vec[i], "O_den": Oden_vec[i], "pressure": Pressure_vec[i],
                "FluxIon": FluxIon_vec[i], "EavgMB": interpolator(Pressure_vec[i]).item(), "current": data_dict["Current"][i], "N_den": N_vec[i]
            })
        
        return np.array(exp_vec, dtype=object), Recomb_prob_exp_vec
    
    
    def MB_func(self, E, TavgMB):
        return ( 2.0/ np.sqrt(np.pi) * ((1.0 / (self.const_dict["kBoltz"] * TavgMB)) ** 1.5)
            * np.exp(-E / (self.const_dict["kBoltz"] * TavgMB)) * (E**0.5))
    
    
    def rates_definition(self, exp_dict, energy_dict):
        
        ### Constants and Experimental Parameters
        F0, S0 = self.const_dict["F0"], self.const_dict["S0"]
        surface = F0 + S0
        
        R = self.const_dict["R"]
        kBoltz = self.const_dict["kBoltz"]
        
        Tw, Tnw = exp_dict["Tw"], exp_dict["Tnw"]
        EavgMB = exp_dict["EavgMB"]
        
        Emin, Ealpha = energy_dict["Emin"], energy_dict["Ealpha"]
        
        ### Energy barriers with the Oxygen Atomic
        E_O_F, E_O_S, E_O_SO, E_O_FO, E_FO_SO, E_FO_FO  = energy_dict["E_O_F"], energy_dict["E_O_S"], energy_dict["E_O_SO"], energy_dict["E_O_FO"], energy_dict["E_FO_SO"], energy_dict["E_FO_FO"]
        E_di_O, E_de_O = energy_dict["E_di_O"], energy_dict["E_de_O"]
        nu_D_oxy1, nu_d_oxy1 = energy_dict["nu_D"], energy_dict["nu_d"]
        
        ### Energy barriers with the Oxygen Molecule
        E_O2_F, E_O2_FO, E_O2_FO2, E_O_FO2, E_FO2_FO = energy_dict["E_O2_F"], energy_dict["E_O2_FO"], energy_dict["E_O2_FO2"], energy_dict["E_O_FO2"], energy_dict["E_FO2_FO"]
        E_di_O2, E_de_O2, E_FO_FO2 = energy_dict["E_di_O2"], energy_dict["E_de_O2"], energy_dict["E_FO_FO2"]
        nu_D_oxy2, nu_d_oxy2 = energy_dict["nu_D"], energy_dict["nu_d"]
        
        
        ### Energy barriers with Oxygen Metastable
        E_O2fast_SO, E_O2fast_S = energy_dict["E_O2fast_SO"], energy_dict["E_O2fast_S"]
        E_O2fast_SOdb, E_O2fast_Sdb, E_Ofast_SOdb, E_Ofast_Sdb = energy_dict["E_O2fast_SOdb"], energy_dict["E_O2fast_Sdb"], energy_dict["E_Ofast_SOdb"], energy_dict["E_Ofast_Sdb"]
        E_O_Sdb, E_O_SOdb, E_FO_SOdb = energy_dict["E_O_Sdb"], energy_dict["E_O_Sdb"], energy_dict["E_FO_SOdb"]
        ED_db = energy_dict["ED_db"]
        
        
        ### SF reactions
        SF_O_F, SF_O_S, SF_O_SO, SF_O_FO = self.SF_dict["SF_O_F"], self.SF_dict["SF_O_S"], self.SF_dict["SF_O_SO"], self.SF_dict["SF_O_FO"]
        SF_FO_SO, SF_FO_S, SF_FO_FO, SF_FO = self.SF_dict["SF_FO_SO"], self.SF_dict["SF_FO_S"], self.SF_dict["SF_FO_FO"], self.SF_dict["SF_FO"]
        
        SF_O2_F, SF_O2_FO, SF_O2_FO2, SF_O_FO2 = self.SF_dict["SF_O2_F"], self.SF_dict["SF_O2_FO"], self.SF_dict["SF_O2_FO2"], self.SF_dict["SF_O_FO2"]
        SF_FO2_FO, SF_FO2, SF_FO_FO2  = self.SF_dict["SF_FO2_FO"], self.SF_dict["SF_FO2"], self.SF_dict["SF_FO_FO2"]
        
        SF_O2fast_SO, SF_Ofast_SO, SF_O2fast_S, SF_Ofast_S = self.SF_dict["SF_O2fast_SO"], self.SF_dict["SF_Ofast_SO"], self.SF_dict["SF_O2fast_S"], self.SF_dict["SF_Ofast_S"]
        SF_Ofast_Sdb, SF_Ofast_SOdb, SF_O2fast_Sdb, SF_O2fast_SOdb = self.SF_dict["SF_Ofast_Sdb"], self.SF_dict["SF_Ofast_SOdb"], self.SF_dict["SF_O2fast_Sdb"], self.SF_dict["SF_O2fast_SOdb"] 
        SF_O_Sdb, SF_O_SOdb = self.SF_dict["SF_O_Sdb"], self.SF_dict["SF_O_SOdb"]
        SF_FO_SOdb, SF_FO_Sdb = self.SF_dict["SF_FO_SOdb"], self.SF_dict["SF_FO_Sdb"]
        
        
        ### Auxiliar quantities
        FluxO = 0.25 * np.sqrt((8.0 * R * 1000 * Tnw)/(0.016 *  np.pi)) * exp_dict["O_den"] * 100
        
        if exp_dict["N_den"] > 0:
            OdenN = exp_dict["N_den"]
        else:
            OdenN = exp_dict["pressure"] * 133.322368 * 1e-6 / (kBoltz * Tnw)
        
        
        OdenO2 = OdenN - exp_dict["O_den"]
        FluxO2 = 0.25 * np.sqrt((8.0 * R * 1000 * Tnw)/(0.032 *  np.pi)) * OdenO2 * 100
        
        flux_dict = {
            "FluxO": FluxO, "FluxO2": FluxO2, "FluxIon": exp_dict["FluxIon"],
        }
        
        TavgMB = EavgMB * 1.60218e-19 / kBoltz
        Emin = Emin * 1.602 * 1e-19
        IntMB = sp.integrate.quad(self.MB_func, Emin, 40*Emin, args=(TavgMB))
        Ealpha = Ealpha * kBoltz
        Intalpha = sp.integrate.quad(self.MB_func, Ealpha, 40*Ealpha, args=(Tnw))
        
        
        ###! Transition rates - Oxigen atomic
        r1 = SF_O_F * FluxO / surface * np.exp(-E_O_F / (R * Tnw))
        r2 = SF_FO * nu_d_oxy1 * np.exp(-E_de_O / (R * Tw))
        r3 = SF_O_S * FluxO / surface * np.exp(-E_O_S / (R * Tnw))
        r4 = SF_O_FO * FluxO / surface * np.exp(-E_O_FO / (R * Tnw))
        r5 = SF_FO_S * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw))
        r6 = SF_O_SO * FluxO / surface * np.exp(-E_O_SO / (R * Tnw))
        r7 = SF_FO_SO * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-E_FO_SO / (R * Tw))
        r8 = SF_FO_FO * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-E_FO_FO / (R * Tw))
        
        rates_oxy_atomic_dict = {
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
            "r5": r5, "r6": r6,"r7": r7, "r8": r8,
        }
        
        
        ###! Transition rates - Oxygen molecular
        r9 = SF_O2_F * FluxO2 / surface * np.exp(-E_O2_F / (R * Tnw))       # same
        r10 = SF_FO2 * nu_d_oxy2 * np.exp(-E_de_O2 / (R * Tw))              # same 
        r11 = SF_O2_FO * FluxO2 / surface * np.exp(-E_O2_FO / (R * Tnw))    # same
        r12 = SF_O2_FO2 * FluxO2 / surface * np.exp(-E_O2_FO2 / (R * Tnw))  # 
        ###* skip r13 for now: O_4(g) -> O_2(g) + O_2(g)
        r14 = SF_O_FO2 * FluxO / surface * np.exp(-E_O_FO2 / (R * Tnw))    # r13 paper Viegas et al. 2024
        r15a = SF_FO2_FO * nu_D_oxy2 * np.exp(-E_di_O2 / (R * Tw)) * np.exp(-E_FO2_FO / (R * Tw)) # r15 paper Viegas et al. 2024
        r15b = SF_FO_FO2 * nu_D_oxy2 * np.exp(-E_di_O2 / (R * Tw)) * np.exp(-E_FO_FO2 / (R * Tw)) # r16 paper Viegas et al. 2024
        
        rates_oxy_molecular_dict = {
            "r9": r9, "r10": r10, "r11": r11, "r12": r12, 
            "r14": r14, "r15a": r15a, "r15b": r15b
        }
        
        ###! Transition rates - Metastable Surface Kinetics
        r16 = SF_O2fast_S * exp_dict["FluxIon"] / surface * IntMB[0] * np.exp(- E_O2fast_S / (R * Tnw))
        ###* correction r16 for the wall temperature: f(Tw), where f(Tw = 50*C) = 1
        r16 = r16 * (
        (np.exp(-E_di_O / (R * Tw)) + np.exp(-E_de_O / (R * Tw)))
        / (np.exp(-E_di_O / (R * 323.15)) + np.exp(-E_de_O / (R * 323.15)))
        )
        r18 = SF_O2fast_SO * exp_dict["FluxIon"] / surface * IntMB[0] * np.exp(- E_O2fast_SO / (R * Tnw))
        r18 = r18 * (
        (np.exp(-E_di_O / (R * Tw)) + np.exp(-E_de_O / (R * Tw)))
        / (np.exp(-E_di_O / (R * 323.15)) + np.exp(-E_de_O / (R * 323.15)))
        )
        
        ###* and the r17 and r19 are assumed to be included in the r16 and r18
        r20 = SF_O_Sdb * FluxO * (1.0 - Intalpha[0]) / surface * np.exp(-E_O_Sdb / (R * Tnw))
        r21 = SF_Ofast_Sdb * FluxO * Intalpha[0] / surface * np.exp(-E_Ofast_Sdb / (R * Tnw))
        
        r22 = SF_O2fast_Sdb * FluxO2 * Intalpha[0] / surface * np.exp(-E_O2fast_Sdb / (R * Tnw))
        r23 = SF_O2fast_SOdb * FluxO2 * Intalpha[0] / surface * np.exp(-E_O2fast_SOdb / (R * Tnw))
        
        r24 = SF_Ofast_SOdb * FluxO * Intalpha[0] / surface * np.exp(-E_Ofast_SOdb / (R * Tnw))
        r25 = SF_O_SOdb * FluxO * (1.0 - Intalpha[0]) / surface * np.exp(- E_O_SOdb / (R * Tnw))
        
        r26 = SF_FO_Sdb * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-ED_db / (R * Tw))
        r27 = SF_FO_SOdb * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-ED_db / (R * Tw)) * np.exp(-E_FO_SOdb / (R * Tw))
        
        rates_metastable_dict = {
            "r16": r16, "r18": r18, "r20": r20, "r21": r21,
            "r22": r22, "r23": r23, "r24": r24, "r25": r25,
            "r26": r26, "r27": r27,
        }
        
        return (rates_oxy_atomic_dict, rates_oxy_molecular_dict, rates_metastable_dict, flux_dict)


    #### Physical system model
    def system_ode(self, X, t, rates_dict_vec):
        
        S0, F0 = self.const_dict["S0"], self.const_dict["F0"]

        frac_Of, frac_O2f, frac_Os, frac_Osdb, frac_Svdb = X
        
        rates_oxy1_dict, rates_oxy2_dict, rates_metastable_dict, flux_dict = rates_dict_vec
        r1, r2, r3, r4, r5, r6, r7, r8 = rates_oxy1_dict.values()
        r9, r10, r11, r12, r14, r15a, r15b = rates_oxy2_dict.values()
        r16, r18, r20, r21, r22, r23, r24, r25, r26, r27 = rates_metastable_dict.values()
        
        FluxO, FluxO2, FluxIon = flux_dict.values()
        
        frac_Fv = 1.0 - frac_Of - frac_O2f
        frac_Sv = 1.0 - frac_Os - frac_Osdb - frac_Svdb
        
        
        frac_Of_equation = [
            + r1 * frac_Fv
            - r2 * frac_Of
            - r4 * frac_Of
            - r5 * frac_Of * (S0/F0) * frac_Sv
            - r7 * frac_Of * (S0/F0) * frac_Os
            - 2.0 * r8 * frac_Of * frac_Of
            
            - r11 * frac_Of
            - (r15a + r15b) * frac_O2f * frac_Of
            
            - r26 * frac_Of * (S0/F0) * frac_Svdb 
            - r27 * frac_Of * (S0/F0) * frac_Osdb
        ]
        
        frac_O2f_equation = [
            + r9 * frac_Fv
            - r10 * frac_O2f
            - r12 * frac_O2f
            - r14 * frac_O2f
            - (r15a + r15b) * frac_O2f * frac_Of
        ]
        
        frac_Os_equation = [
            + r3 * frac_Sv
            - r6 * frac_Os
            + r5 * frac_Of * frac_Sv
            - r7 * frac_Of * frac_Os
            - r16 * frac_Os
        ]
        
        frac_Osdb_equation = [
            + r20 * frac_Svdb
            - r23 * frac_Osdb
            - r24 * frac_Osdb
            - r25 * frac_Osdb 
            + r26 * frac_Of * frac_Svdb
            - r27 * frac_Of * frac_Osdb
        ]
        
        frac_Svdb_equation = [
            + r16 * frac_Sv
            + r18 * frac_Os
            - r20 * frac_Svdb
            - r21 * frac_Svdb
            - r22 * frac_Svdb #-
            + r25 * frac_Osdb
            - r26 * frac_Of * frac_Svdb
            + r27 * frac_Of * frac_Osdb
        ]
        
        
        func = [frac_Of_equation[0], frac_O2f_equation[0], frac_Os_equation[0], frac_Osdb_equation[0], frac_Svdb_equation[0]]
        return func    
    
    
    ### Solvers
    def solve_ode(self, rates_dict, method="stiff"):
        if method == "simple":
            timeSpace = np.linspace(0, 10_000, 1_000)
            solution = sp.integrate.odeint(
                func=self.system_ode,
                y0=self.init_conditions,
                t=timeSpace,
                args=(rates_dict,)
            )
        elif method == "stiff":
            sol = sp.integrate.solve_ivp(
                fun=lambda t, X: self.system_ode(X, t, rates_dict),
                t_span=(self.timeSpace[0], self.timeSpace[-1]),
                y0=self.init_conditions,
                method="BDF",
                t_eval=self.timeSpace,
                atol=1e-5, rtol=1e-5
            )
            solution = sol.y.T
        else:
            raise ValueError("Invalid method - choose between 'simple' and 'stiff'")
        
        sucess = self.solution_check(solution, rates_dict)
        
        return solution, sucess
    
    
    def solve_fixed_point(self, rates_dict, max_time=15):
        
        ### Improve initial guess by running using the stiff solver
        #### BDF method
        
        # short_time = np.linspace(self.timeSpace[0], self.timeSpace[min(100, len(self.timeSpace)-1)], 100)
        short_time = np.linspace(self.timeSpace[0], self.timeSpace[min(30, len(self.timeSpace)-1)], 30)
        
        start_time = time.time()
        events = None
        if max_time is not None:
            def timeout_event(t, X):
                elapsed = time.time() - start_time
                return max_time - elapsed
            
            timeout_event.terminal = True
            timeout_event.direction = -1
            events = [timeout_event]
            
        
        sol_short = sp.integrate.solve_ivp(
            fun=lambda t, X: self.system_ode(X, t, rates_dict),
            t_span=(short_time[0], short_time[-1]),
            y0=self.init_conditions,
            method="Radau",
            t_eval=short_time,
            atol=1e-5, rtol=1e-5,
            events=events
        )
        
        refined_guess = sol_short.y.T[-1]
        
        ### Attempt to find the fixed point using the refined guess
        try:
            sol = sp.optimize.root(self.system_ode, refined_guess, args=(0, rates_dict), method="hybr")
            success = self.solution_check(np.atleast_2d(sol.x), rates_dict)
        except Exception as e:
            print("Fixed point solver failed with message: ", e)
            success = False
            sol = self.init_conditions
        
        return sol, success
    
    
    def solution_check(self, sol, rates_dict):
        vec_aux = self.system_ode(sol[-1], 0, rates_dict)
        absolute_error = np.sum(np.abs(vec_aux))
        # print("Absolute error: ", absolute_error)
        if absolute_error > 1e-4:
            return False
        else:
            return True
    
    
    ### Recombination Probability Computations
    def compute_gammas(self, frac_solutioms, rates_dict):
        
        S0, F0 = self.const_dict["S0"], self.const_dict["F0"]
        frac_Ofss, frac_O2fss, frac_Osss, frac_Osdbss, frac_Svdbss = frac_solutioms
        
        rates_oxy_atomic_dict, rates_oxy_molecular_dict, rates_metastable_dict, flux_dict = rates_dict
        
        r1, r2, r3, r4, r5, r6, r7, r8 = rates_oxy_atomic_dict.values()
        r9, r10, r11, r12, r14, r15a, r15b = rates_oxy_molecular_dict.values()
        r16, r18, r20, r21, r22, r23, r24, r25, r26, r27 = rates_metastable_dict.values()
        
        FluxO, FluxO2, FluxIon = flux_dict.values()
        
        ###! Atomic Oxygen
        gamma_r4 = 2.0 * r4 * frac_Ofss * F0 / FluxO
        gamma_r6 = 2.0 * r6 * frac_Osss * S0 / FluxO
        gamma_r7 = 2.0 * r7 * frac_Osss * S0 * frac_Ofss / FluxO
        gamma_r8 = 2.0 * r8 * frac_Ofss * F0 * frac_Ofss / FluxO
        
        ###! Molecular Oxygen
        
        gamma_r11 = r11 * frac_Ofss * F0 / FluxO
        gamma_r12 = r12 * frac_O2fss * F0 / FluxO
        gamma_r14 = r14 * frac_Ofss * F0 / FluxO
        gamma_r15 = (r15a + r15b) * frac_O2fss * F0 * frac_Ofss / FluxO
        
        ####! Metastable Surface Kinetics
        gamma_r23 = 2.0 * r23 * frac_Osdbss * S0 / FluxO
        gamma_r24 = 2.0 * r24 * frac_Osdbss * S0 / FluxO
        gamma_r25 = 2.0 * r25 * frac_Osdbss * S0 / FluxO
        gamma_r27 = 2.0 * r27 * frac_Osdbss * S0 * frac_Ofss / FluxO
        
        
        results_gammas = [gamma_r4, gamma_r6, gamma_r7, gamma_r8,\
                        gamma_r11, gamma_r12, gamma_r14, gamma_r15,\
                        gamma_r23, gamma_r24, gamma_r25, gamma_r27]
        
        results_names = ["g_r4", "g_r6", "g_r7", "g_r8",\
                        "g_r11", "g_r12", "g_r14", "g_r15",\
                        "g_r23", "g_r24", "g_r25", "g_r27"] 
        
        return results_gammas, results_names
    
    
    def solve_system(self, exp_dict, energy_dict, solver="fixed_point", max_time=None):
        rates_dict = self.rates_definition(exp_dict, energy_dict)
        
        if solver == "odeint":
            sol, success = self.solve_ode(rates_dict)
            steady_state_sol = tuple(sol[-1])
            
            if success == False:
                print("Failed to find the steady state solution")
                print("Exp dict: ", exp_dict)
            
        elif solver == "fixed_point":
            
            steady_state_sol, success  = self.solve_fixed_point(rates_dict, max_time=max_time)
            if success == False:
                print("Failed to find the steady state solution")
                print("Exp dict: ", exp_dict)
            
            steady_state_sol = tuple(steady_state_sol.x)
        else:
            raise ValueError("Invalid solver")
        
        results_gammas, results_names = self.compute_gammas(steady_state_sol, rates_dict)
        
        return steady_state_sol, results_gammas, results_names, success
