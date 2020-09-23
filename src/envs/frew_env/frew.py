import win32com.client
import sys
import shutil, os
import datetime
import time
import math
import numpy as np

class Frew():


    def __init__(self, base_model_path):
        if not os.path.exists(base_model_path):
            raise Exception('Frew Base Model cannot be located. Please verify file path.')

        # initialize frew model and delete past analyses
        self.model = win32com.client.Dispatch("frewLib.FrewComAuto")
        self.model.Open(base_model_path)
        self.model.DeleteResults()  

        # get node and stage count from model 
        self.node_count= self.model.GetNumNodes()
        self.stage_count = self.model.GetNumStages()

        self.node_interval = 0

        #run pre-checks
        self.pre_checks()

        self.sim_results = {}
    

    def close_model(self):
        self.model.Close()


    def reset(self):
        self.model.DeleteResults()


    def pre_checks(self):
        for i in range(1, self.node_count):
            node_level1 = self.model.GetNodeLevel(i)
            node_level2 = self.model.GetNodeLevel(i-1)
            if i>1:
                if node_level2  -node_level1 != node_int_check:
                    self.model.Close()
                    raise Exception('ERROR: please ensure consistent nodal intervals are adopted in the base model.\nERROR: SIMULATION ABORTED')
            else:
                node_int_check = node_level2 - node_level1

        node_interval = node_int_check
        wall_top_level = self.model.GetNodeLevel(0)
        print('The nodal interval of ' + str(node_interval) + 'm has been adopted in the base model.')
        print('The top of the wall has been specified as ' + str(wall_top_level) + 'm')

        excavation_depth_max_node=0
        for i in range(0, self.stage_count):
            for k in range(0, self.node_count):
                if self.model.GetSoilZoneRight(k,i) == 0 and k > excavation_depth_max_node:
                    excavation_depth_max_node=k

        excavation_depth_max = wall_top_level  - (self.model.GetNodeLevel(excavation_depth_max_node) + self.model.GetNodeLevel(excavation_depth_max_node+1))/2
        self.node_interval = node_interval
        print('The maximum excavation depth specified in the base model is ' + str(excavation_depth_max)+'m')

    def cantilever_analysis(self, 
                    wall_depth_sim,
                    pile_diameter,  
                    pile_spacing
                    ): 

        #Calculation of EI and Creation of the EI Maxtrix
        secant_wall_E=28000000 
        pile_I=math.pi*(pile_diameter**4)/64
        wall_EI=secant_wall_E*pile_I/pile_spacing
        pile_setup=[pile_diameter,pile_spacing,wall_EI]
    
        deflection_criteria_satisfied=True
        BM_criteria_satisfied=True
        SF_criteria_satisfied=True
        
        wall_EI_sim = pile_setup[2]
        wallbase_node_sim = math.ceil(wall_depth_sim/self.node_interval)
        for i in range (1, self.stage_count):
            for k in range(0, self.node_count):
                if k <= wallbase_node_sim:
                    self.model.SetWallEI(k, i, wall_EI_sim)
                else:
                    self.model.SetWallEI(k, i, 0)

        self.model.DeleteResults() #results MUST be deleted before running an analysis. Else a 'com_error' will occur.
        self.model.Analyse(self.stage_count)

        #Verify if converged solutions have been obtained at ALL stages (no results will be returned if the analysis failed to converge at all stages)
        if self.model.GetNumStages() < self.stage_count:
            raise Exception('Frew analysis failed to converge at stage ' + str(self.model.GetNumStages() + 1))

        max_deflection = 0
        max_BM = 0
        max_SF = 0
        for i in range(0, self.stage_count):
            for k in range(0, self.node_count):
                node_deflection_temp = self.model.GetNodeDisp(k,i)
                node_BM_temp = self.model.GetNodeBending(k,i)
                node_SF_temp = self.model.GetNodeShear(k,i)

                if abs(node_deflection_temp) > abs(max_deflection):
                    max_deflection = node_deflection_temp
                if abs(node_BM_temp) > abs(max_BM):
                    max_BM = node_BM_temp        
                if abs(node_SF_temp) > abs(max_SF):
                    max_SF = node_SF_temp

        self.sim_results = {
            'max_deflection': round(max_deflection,2),
            'max_bm': round(max_BM,2),
            'max_sf': round(max_SF,2)
        }

        return self.sim_results

    def validate_results(self,
                    deflection_limit,
                    BM_limit,
                    SF_limit,
                    deflection_criteria_switch=True,
                    BM_criteria_switch=False,
                    SF_criteria_switch=False):

        validation = {
            'deflection': True,
            'bm': True,
            'sf': True
        }
    
        if deflection_criteria_switch == True and abs(self.sim_results['max_deflection']) > deflection_limit:
            validation['deflection']=False

        if BM_criteria_switch == True and abs(self.sim_results['max_BM']) > BM_limit:
            validation['bm'] = False       

        if SF_criteria_switch == True and abs(self.sim_results['max_SF']) > SF_limit:
            validation['sf'] = False

        return validation  

if __name__ == '__main__':
    base_model_path=r"C:\projects\frw-rl\models\CANTTEST.fwd"

    # parameters to be optimized
    wall_depth=20
    pile_diameter=2
    pile_spacing=2.5

    # design criteria
    deflection_limit=1000 #in mm
    BM_limit=3000 #in kNm/m
    SF_limit=600 #in kN/m

    # initialize frew model
    frw = Frew(base_model_path)

    # execute the analysis in Frew
    sim_results = frw.cantilever_analysis(
                                    wall_depth,
                                    pile_diameter,
                                    pile_spacing
                                    )

    # verify if all conditions are met
    validation = frw.validate_results(
                        deflection_limit,
                        BM_limit,
                        SF_limit
                        )

    if not all(validation.values()):
        not_satisfied = ', '.join([crit for crit in validation.keys() if not validation[crit]])
        print(f'{not_satisfied} criteria not satisfied!')
    else: 
        print('The Max Deflection, Max BM and Max SF are ' \
        f'{sim_results["max_deflection"]}mm, ' \
        f'{sim_results["max_bm"]}gkNm/m and ' \
        f'{sim_results["max_sf"]}gkN/m, respectively.')

    