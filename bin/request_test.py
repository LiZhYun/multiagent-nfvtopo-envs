from __future__ import division
import sys
import pickle   
import random
import numpy, numpy.random
import pprint
from datetime import datetime
from multiagentnfv.core_nfv import Request as Req
from multiagentnfv.parser import Parser
from itertools import permutations, islice, product, chain
from copy import deepcopy



def requestlist():
        request_list = set()

        ### abilene ###
        ###############
        reqtest = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2],'u3':[4.0]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 3,('a1','a2_1'): 4,('a1','a2_2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1{u1;a2;3}"}



        ### Mixed request set ###
        #########################

        req140 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
                'A': {'a1': 9, 'a2': 2},
                'l_req' : {('a1','a2'): 4},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}
                
        req150 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
                'A': {'a1': 2, 'a2': 9},
                'l_req' : {('a1','a2'): 5.5},
                'input_datarate' : 100,
                'chain' : "a1.u1.u2.u3.a2"}     
                
        # req160 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        #         'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        #         'A': {'a1': 9, 'a2': 3},
        #         'l_req' : {('a1','a2'): 7},
        #         'input_datarate' : 100,
        #         'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"} 

        req160 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 0, 'a2': 5},
                'l_req' : {('a1','a2'): 7},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"} 

        req161 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 9, 'a2': 8},
                'l_req' : {('a1','a2'): 7},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"}  

        req162 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 2, 'a2': 7},
                'l_req' : {('a1','a2'): 7},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"}         


        req170 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2],'u3':[4.0]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 3,('a1','a2_1'): 4,('a1','a2_2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1{u1,u2,u3;a2;3}"}

        req171 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.7],'u3':[4.0]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 6,('a1','a2_1'): 5,('a1','a2_2'): 7},
                'input_datarate' : 200,
                'chain' : "a1.u1{u1,u2,u3;a2;3}"}

        req172 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.25],'u2':[0.1],'u3':[3.0]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 8,('a1','a2_0'): 5,('a1','a2_1'): 3,('a1','a2_2'): 5},
                'input_datarate' : 150,
                'chain' : "a1.u1{u1,u2,u3;a2;3}"}
                
        req180 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.1]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 6,('a1','a2_1'): 5.5,('a1','a2_2'): 5},
                'input_datarate' : 200,
                'chain' : "a1.u1{u1,u2;a2;3}"}  
                
        req181 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.7]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 4,('a1','a2_0'): 5.5,('a1','a2_1'): 5,('a1','a2_2'): 7.5},
                'input_datarate' : 150,
                'chain' : "a1.u1{u1,u2;a2;3}"}      

        ### Broadband network ###
        #########################

        # tenant1
        req00 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
                'A': {'a1': 9, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}
        req01 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
                'A': {'a1': 4, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}
        req02 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
                'A': {'a1': 10, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}
        req03 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
                'A': {'a1': 3, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}

        req10 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
                'A': {'a1': 2, 'a2': 8},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 700,
                'chain' : "a1.u1.u2.u3.a2"}
        req11 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
                'A': {'a1': 2, 'a2': 4},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.u1.u2.u3.a2"}
        req12 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
                'A': {'a1': 2, 'a2': 10},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.u1.u2.u3.a2"}
        req13 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
                'A': {'a1': 2, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.u1.u2.u3.a2"}

        # tenant 2
        req20 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
                'A': {'a1': 3, 'a2': 4},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 150,
                'chain' : "a1.u1.u2.a2"}
        req21 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
                'A': {'a1': 7, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1.u2.a2"}
        req22 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
                'A': {'a1': 11, 'a2': 9},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 150,
                'chain' : "a1.u1.u2.a2"}
        req23 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
                'A': {'a1': 10, 'a2': 2},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 150,
                'chain' : "a1.u1.u2.a2"}


        req30 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
                'A': {'a1': 2, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.(u1,u2).u3.a2"}
        req31 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
                'A': {'a1': 2, 'a2': 7},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.(u1,u2).u3.a2"}
        req32 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
                'A': {'a1': 2, 'a2': 11},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.(u1,u2).u3.a2"}
        req33 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
                'A': {'a1': 2, 'a2': 1},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 10,
                'chain' : "a1.(u1,u2).u3.a2"}
                
        ### Mobile core network ###
        ###########################

        # tenant 3
        req40 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 9, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
        req41 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.2,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 4, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
        req42 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.2,0.5,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 1, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
        req43 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.2,0.5,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
                'A': {'a1': 8, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     

        req50 = {'UF' : {'a1':'WWW', 'a2':'GGSN','a3':'GGSN','a4':'GGSN','a5':'GGSN','u1':'VOPT','u2':'DPI'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a3':[1.0],'a4':[1.0],'a5':[1.0],'u1':[2.0],'u2':[0.25,0.25,0.25,0.25]},
                'A': {'a1': 3, 'a2': 9, 'a3': 4, 'a4': 1, 'a5': 8},
                'l_req' : {('a1','a2'):5,('a1','a3'):5,('a1','a4'):5,('a1','a5'):5},
                'input_datarate':100,
                'chain' : "a1.u1.u2[a2|a3|a4|a5]"}

        req51 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'VOPT','u2':'DPI', 'u3':'FW', 'u4':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[2.0],'u2':[0.5,0.5], 'u3':[1.0], 'u4':[1.0]},
                'A': {'a1': 7, 'a2': 11},
                'l_req' : {('a1','a2'):5},
                'input_datarate':200,
                'chain' : "a1.u1.u2[u3|u4].a2"}

        req52 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'VOPT','u2':'DPI', 'u3':'FW', 'u4':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[2.0],'u2':[0.5,0.5], 'u3':[1.0], 'u4':[1.0]},
                'A': {'a1': 0, 'a2': 7},
                'l_req' : {('a1','a2'):5},
                'input_datarate':200,
                'chain' : "a1.u1.u2[u3|u4].a2"}

        # tenant 4
        req60 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
                'A': {'a1': 10, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 5,
                'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
        req61 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
                'A': {'a1': 9, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 5,
                'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
        req62 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
                'A': {'a1': 0, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 5,
                'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
        req63 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
                'A': {'a1': 3, 'a2': 3},
                'l_req' : {('a1','a2'): 5},
                'input_datarate' : 5,
                'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        

        req70 = {'UF' : {'a1':'WWW', 'a2':'GGSN','a3':'GGSN','a4':'GGSN','a5':'GGSN','u1':'DPI','u2':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a3':[1.0],'a4':[1.0],'a5':[1.0],'u1':[1.0],'u2':[2.0]},
                'A': {'a1': 6, 'a2': 10},
                'l_req' : {('a1','a2'):5},
                'input_datarate':5,
                'chain' : "a1.u1.u2.a2"}
        req71 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
                'A': {'a1': 11, 'a2': 9},
                'l_req' : {('a1','a2'):5},
                'input_datarate':5,
                'chain' : "a1.u1.u2.a2"}
        req72 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
                'A': {'a1': 5, 'a2': 0},
                'l_req' : {('a1','a2'):5},
                'input_datarate':5,
                'chain' : "a1.u1.u2.a2"}
        req73 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
                'A': {'a1': 7, 'a2': 8},
                'l_req' : {('a1','a2'):5},
                'input_datarate':5,
                'chain' : "a1.u1.u2.a2"}

        ### Data center ###
        ###################

        # tenant 5
        req80 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.8]},
                'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
                'input_datarate' : 200,
                'chain' : "a1.u1{u1,u2;a2;3}"}
        req81 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2]},
                'A': {'a1': 4,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
                'input_datarate' : 100,
                'chain' : "a1.u1{u1,u2;a2;3}"}
        req82 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.4]},
                'A': {'a1': 8,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
                'input_datarate' : 150,
                'chain' : "a1.u1{u1,u2;a2;3}"}
        req83 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.2],'u2':[0.8]},
                'A': {'a1': 9,'a2_0': 0,'a2_1': 7,'a2_2': 10},
                'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
                'input_datarate' : 200,
                'chain' : "a1.u1{u1,u2;a2;3}"}

        ### Complex Request ###
        #######################

        req9 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'WOPT','u3':'AV','u4':'DPI','u5':'WAPGW','u6':'PCTL','u7':'FW','u8':'VOPT','u9':'CACHE','u10':'IDS','u11':'IPS'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'a2_3':[1.0],'u1':[0.25],'u2':[1.5],'u3':[0.75],'u4':[0.25,0.1,0.6,0.05],'u5':[1.0],'u6':[1.4],'u7':[0.8],'u8':[2.5],'u9':[1.0],'u10':[0.75],'u11':[0.9]},
                'A' : {'a1': 4 ,'a2_0': 10,'a2_1': 7,'a2_2': 0,'a2_3' : 8},
                'l_req' : {('a1','a2'): 10,('a1','a2_0'): 10,('a1','a2_1'): 10,('a1','a2_2'): 10,('a1','a2_3'): 10},
                'input_datarate' : 150,
                'chain' : "a1.u1{u1,u2,u3;u4[u5|(u6,u7)|u8.u9|u10].u11.a2;4}"}
                
        ### Optional Order ###
        ######################

        req100 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'VOPT','u3':'FW'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[4.0],'u3':[0.4]},
                'A' : {'a1': 9, 'a2': 8},
                'l_req' : {('a1','a2'): 1},
                'input_datarate' : 150,
                'chain' : "a1.(u1,u2,u3).a2"}       
                
        req110 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'WOPT','u3':'AV','u4':'DPI','u5':'WAPGW','u6':'FW'}, 
                'r' : {'a1':[1.0],'a2':[1.0],'u1':[0.3],'u2':[2.5],'u3':[0.8],'u4':[1.0],'u5':[1.0],'u6':[0.4]},
                'A' : {'a1': 3 ,'a2': 0},
                'l_req' : {('a1','a2'): 1},
                'input_datarate' : 200,
                'chain' : "a1.u1{u1,u2,u3;u4.u5;3}.u6.a2"}

        req120 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'FW','u2':'VOPT'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.8],'u2':[2.0]},
                'A' : {'a1': 1, 'a2': 10},
                'l_req' : {('a1','a2'): 1},
                'input_datarate' : 100,
                'chain' : "a1.(u1,u2).a2"}
                
        req130 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE','u3':'AV','u4':'DPI','u5':'VOPT'}, 
                'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5],'u2':[1.0],'u3':[0.5],'u4':[1.0],'u5':[3.0]},
                'A' : {'a1': 11, 'a2': 9},
                'l_req' : {('a1','a2'): 1},
                'input_datarate' : 150,
                'chain' : "a1.(u1,u2).u3.(u4,u5).a2"}
                


        request_list.add(Request(req140))
        request_list.add(Request(req150))
        request_list.add(Request(req160))
        request_list.add(Request(req170))
        request_list.add(Request(req171))
        request_list.add(Request(req172))
        request_list.add(Request(req180))
        request_list.add(Request(req181))

        request_list.add(Request(req150))

        request_list.add(Request(req20))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req50))
        request_list.add(Request(req51))
        request_list.add(Request(req52))
        request_list.add(Request(req160))
        request_list.add(Request(req161))
        request_list.add(Request(req162))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))    
        request_list.add(Request(req73))    
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req50))
        request_list.add(Request(req51))
        request_list.add(Request(req52))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))
        request_list.add(Request(req73))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req50))
        request_list.add(Request(req51))
        request_list.add(Request(req52))
        request_list.add(Request(req160))
        request_list.add(Request(req161))
        request_list.add(Request(req162))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))    
        request_list.add(Request(req73))    
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req50))
        request_list.add(Request(req51))
        request_list.add(Request(req52))
        request_list.add(Request(req160))
        request_list.add(Request(req161))
        request_list.add(Request(req162))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))    
        request_list.add(Request(req73))    
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req50))
        request_list.add(Request(req51))
        request_list.add(Request(req52))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))
        request_list.add(Request(req73))
        request_list.add(Request(req160))
        #~ request_list.add(Request(req140))
        #~ request_list.add(Request(req150))
        #~ request_list.add(Request(req160))
        request_list.add(Request(req170))
        #~ request_list.add(Request(req171))
        #~ request_list.add(Request(req172))
        #~ request_list.add(Request(req180))
        #~ request_list.add(Request(req181))

        #~ request_list.add(Request(req140))
        #~ request_list.add(Request(req150))
        #~ request_list.add(Request(req160))
        request_list.add(Request(req170))
        #~ request_list.add(Request(req171))
        #~ request_list.add(Request(req172))
        request_list.add(Request(req180))
        #~ request_list.add(Request(req181))

        #~ request_list.add(Request(req140))
        #~ request_list.add(Request(req150))
        #~ request_list.add(Request(req160))
        request_list.add(Request(req170))
        request_list.add(Request(req171))
        request_list.add(Request(req172))
        #~ request_list.add(Request(req180))
        #~ request_list.add(Request(req181))

        #~ request_list.add(Request(req140))
        #~ request_list.add(Request(req150))
        #~ request_list.add(Request(req160))
        #~ request_list.add(Request(req170))
        #~ request_list.add(Request(req171))
        #~ request_list.add(Request(req172))
        request_list.add(Request(req180))
        request_list.add(Request(req181))

        request_list.add(Request(req140))
        #~ request_list.add(Request(req150))
        #~ request_list.add(Request(req160))
        request_list.add(Request(req170))
        #~ request_list.add(Request(req171))
        #~ request_list.add(Request(req172))
        request_list.add(Request(req180))
        #~ request_list.add(Request(req181))

        request_list.add(Request(req10))

        request_list.add(Request(req50))

        request_list.add(Request(req51))

        request_list.add(Request(req52))

        request_list.add(Request(req80))    

        request_list.add(Request(req00))
        request_list.add(Request(req01))
        request_list.add(Request(req02))
        request_list.add(Request(req03))
        request_list.add(Request(req10))
        request_list.add(Request(req11))
        request_list.add(Request(req12))
        request_list.add(Request(req13))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req30))
        request_list.add(Request(req31))
        request_list.add(Request(req32))
        request_list.add(Request(req33))

        request_list.add(Request(req00))
        request_list.add(Request(req01))
        request_list.add(Request(req02))
        request_list.add(Request(req03))
        request_list.add(Request(req10))
        request_list.add(Request(req11))
        request_list.add(Request(req12))
        request_list.add(Request(req13))

        request_list.add(Request(req00))
        request_list.add(Request(req01))
        request_list.add(Request(req02))
        request_list.add(Request(req03))

        request_list.add(Request(req40))
        request_list.add(Request(req41))
        request_list.add(Request(req42))
        request_list.add(Request(req43))
        request_list.add(Request(req50))
        request_list.add(Request(req60))
        request_list.add(Request(req61))
        request_list.add(Request(req62))
        request_list.add(Request(req63))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))
        request_list.add(Request(req73))

        request_list.add(Request(req40))
        request_list.add(Request(req41))
        request_list.add(Request(req42))
        request_list.add(Request(req43))
        request_list.add(Request(req50))

        request_list.add(Request(req40))
        request_list.add(Request(req41))
        request_list.add(Request(req42))

        request_list.add(Request(req80))
        request_list.add(Request(req81))
        request_list.add(Request(req82))
        request_list.add(Request(req83))
        request_list.add(Request(req80))
        request_list.add(Request(req81))
        #~ request_list.add(Request(req82))
        #~ request_list.add(Request(req83))  
        #~ request_list.add(Request(req110)) 
        request_list.add(Request(req130))    
        request_list.add(Request(req170))    
        request_list.add(Request(req80))
        request_list.add(Request(req81))

        request_list.add(Request(req9))

        request_list.add(Request(req00))
        request_list.add(Request(req01))
        request_list.add(Request(req02))
        request_list.add(Request(req03))
        request_list.add(Request(req10))
        request_list.add(Request(req11))
        request_list.add(Request(req12))
        request_list.add(Request(req13))
        request_list.add(Request(req20))
        request_list.add(Request(req21))
        request_list.add(Request(req22))
        request_list.add(Request(req23))
        request_list.add(Request(req30))
        request_list.add(Request(req31))
        request_list.add(Request(req32))
        request_list.add(Request(req33))
        request_list.add(Request(req40))
        request_list.add(Request(req41))
        request_list.add(Request(req42))
        request_list.add(Request(req43))
        request_list.add(Request(req50))
        request_list.add(Request(req60))
        request_list.add(Request(req61))
        request_list.add(Request(req62))
        request_list.add(Request(req63))
        request_list.add(Request(req70))
        request_list.add(Request(req71))
        request_list.add(Request(req72))
        request_list.add(Request(req73))
        request_list.add(Request(req80))
        request_list.add(Request(req81))
        request_list.add(Request(req82))
        request_list.add(Request(req83))

        request_list.add(Request(req100))
        request_list.add(Request(req110))
        request_list.add(Request(req120))
        request_list.add(Request(req130))

        request_list.add(Request(req100))

        request_list.add(Request(req110))

        request_list.add(Request(req120))
        request_list.add(Request(req130))
        request_list.add(Request(req140))  

        request_list.add(Request(req150))    

        request_list.add(Request(req100))
        request_list.add(Request(req110))

        request_list.add(Request(req100))
        request_list.add(Request(req120))

        request_list.add(Request(req100))
        request_list.add(Request(req130))

        request_list.add(Request(req100))
        request_list.add(Request(req110))
        request_list.add(Request(req120))

        request_list.add(Request(req110))
        request_list.add(Request(req120))

        request_list.add(Request(req110))

        request_list.add(Request(req110))
        request_list.add(Request(req120))
        request_list.add(Request(req130))

        request_list.add(Request(req120))
        request_list.add(Request(req130))

        request_list.add(Request(req100))
        request_list.add(Request(req110))
        request_list.add(Request(req130))

        request_list.add(Request(reqtest))

        
        num_requests = len(request_list)
        numpy.random.seed(0)
        ratios = numpy.random.dirichlet(numpy.ones(num_requests),size=1)[0]
        for i,r in enumerate(request_list):
                r.input_datarate = round(ratios[i]*1000,3)
                for k in r.A.keys():
                        r.A[k] = numpy.random.randint(0,11)
        
        return list(request_list)
        # print r
# pickle.dump({"name": reqtype, "reqs": request_list, "seed": seed}, open("requestList_" + reqtype + ".pickle","wb"))


def create_request(request_n, agents):
    for index, request in enumerate(request_n):
        if request == 0:
            agents[index].request = None
            agents[index].processing = False
        else:
            agents[index].processing = True
            new_request = Req()
            new_request.ID = request.prefix
            new_request.optord_vnfs = request.optords
            for key, opts in new_request.optord_vnfs.items():
                for opt_index in range(len(opts)):
                    opts[opt_index] = '_'.join(opts[opt_index].split('_')[1:])
            new_request.request_class = request
            new_request.arrival_time = datetime.now()
            new_request.reliability_requirements = numpy.random.random() * 0.15 + 0.8
            agents[index].request = new_request


def process_request(request_n, REQ_NUM):
    train_vnf_n = []
    tran_pos_n = []
    for request in request_n:
        if request == 0:
            train_vnf_n.append(0)
            tran_pos_n.append(0)
            continue
        request.add_prefix(REQ_NUM)
        REQ_NUM += 1
        request.forceOrder = dict()
        prsr = Parser(request)
        prsr.preparse()
        parseDict = prsr.parseDict.copy()
        optords = prsr.optorderdict.copy()
        request.optords = optords
        parallel_num = prsr.parallel_num.copy()
        perms = dict()
        for v in optords.values():
            perms[",".join(v)] = []
            for x in permutations(v):
                perms[",".join(v)].append(x)

        prod = list(product(*perms.values()))[0]
        reqPlacementInputList = []   # 根据所有optorder可能的顺序组合  构造每个确定顺序的排列
        for i in range(len(prod)):
            for k in optords.keys():
                if prod[i] in perms[k]:
                    request.forceOrder[k] = prod[i]
        prsr = Parser(request)
        prsr.parse()
        # prsr.fixOptionalOrders(0, len(req.chain))
        # prsr.fixNexts(len(req.chain), 0, len(req.chain))
        reqPlacementInput = prsr.create_pairs()
        posencode = posEncode(reqPlacementInput, parseDict, optords)
        transformer_input = {}
        for k, v in posencode.items():
            for m, n in reqPlacementInput['UF'].items():
                if k == m:
                    transformer_input[k] = []
                    transformer_input[k].append(posencode[k])
                    transformer_input[k].append(reqPlacementInput['UF'][k])
                    posencode[k] = transformer_input[k]
                    reqPlacementInput['UF'][k] = transformer_input[k]
                else:
                    transformer_input[k] = posencode[k]
                    transformer_input[m] = reqPlacementInput['UF'][m]
        for k in list(transformer_input.keys()):
            if type(transformer_input[k]) != list:
                del transformer_input[k]
        train_vnf = ''
        tran_pos = ''
        for k, v in transformer_input.items():
            train_vnf += v[1] + ' '
            tran_pos += str(v[0]) + ' '
        train_vnf_n.append(train_vnf)
        tran_pos_n.append(tran_pos)
    return REQ_NUM


def posEncode(reqPlacementInput, parseDict, optords):
    funclist = []
    previsopt = False
    posencode = {}
    lastpair = ''
    firstpair = True
    for key, value in optords.items():
        funclist.extend(value)
    for pair in reqPlacementInput['U_pairs']:
        if firstpair:
            posencode[pair[0]] = 0
            lastpair = pair[0]
            firstpair = False
            # 此对的前一个在optlist中
            if pair[0] in funclist or '_'.join(pair[0].split('_')[:-1]) in funclist:
                previsopt = True
            else:  # 第一个不在optlist中，则不论上一对的最后是否在optlist中，其pos均为上一对后一个的pos+1
                previsopt = False
        if posencode.get(pair[0], None) == None:
            if posencode.get('_'.join(pair[0].split('_')[:-1]) + '_0', None) != None:
                # posencode[pair[0]] = posencode['_'.join(pair[0].split('_')[:-1]) + '_0']
                continue
            # 此对的前一个在optlist中
            if pair[0] in funclist or '_'.join(pair[0].split('_')[:-1]) in funclist:
                if previsopt == True:  # 如果此对的前一对的后一个在optlist中，则此对前一个的pos与上一个相同
                    posencode[pair[0]] = posencode[lastpair]
                else:  # 否则为+1
                    posencode[pair[0]] = posencode[lastpair] + 1
                previsopt = True
            else:  # 第一个不在optlist中，则不论上一对的最后是否在optlist中，其pos均为上一对后一个的pos+1
                posencode[pair[0]] = posencode[lastpair] + 1
                previsopt = False
        if posencode.get(pair[1], None) != None:
            if posencode.get('_'.join(pair[1].split('_')[:-1]) + '_0', None) != None:
                if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                    if previsopt == True:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]]:
                            posencode[pair[1]] = posencode['_'.join(
                                pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]]
                    else:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                            posencode[pair[1]] = posencode['_'.join(
                                pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = True
                else:
                    if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                        posencode[pair[1]] = posencode['_'.join(
                            pair[1].split('_')[:-1]) + '_0']
                    else:
                        posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = False
                continue
        if posencode.get(pair[1], None) == None:
            if posencode.get('_'.join(pair[1].split('_')[:-1]) + '_0', None) != None:
                if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                    if previsopt == True:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]]:
                            posencode[pair[1]] = posencode['_'.join(
                                pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]]
                    else:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                            posencode[pair[1]] = posencode['_'.join(
                                pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = True
                else:
                    if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                        posencode[pair[1]] = posencode['_'.join(
                            pair[1].split('_')[:-1]) + '_0']
                    else:
                        posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = False
                continue
            if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                if previsopt == True:  # 如果此对的前一个在optlist中，则此对前一个的pos与上一个相同
                    posencode[pair[1]] = posencode[pair[0]]
                else:  # 否则为+1
                    posencode[pair[1]] = posencode[pair[0]] + 1
                previsopt = True
            else:
                posencode[pair[1]] = posencode[pair[0]] + 1
                previsopt = False
        lastpair = pair[1]
    return posencode

class Request:
        def __init__(self, req):
                # Usage requests for an instance of a VNF type and the VNF type they require
                # {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}
                self.UF = deepcopy(req['UF'])

                # Usage requests
                self.U = deepcopy(list(req['UF'].keys()))  # a1 a2 u1 u2
                # Ratio of the outgoing data rate to incoming data rate for each branch
                # of each VNF instance that is requested
                self.r = deepcopy(req['r'])

                # Application tier instances and the nodes they're mapped to
                self.A = deepcopy(req['A'])

                # Maximum acceptable latency between the application tiers
                self.l_req = deepcopy(req['l_req'])

                # Incoming data rate to the entrance point of the requested chain
                self.input_datarate = deepcopy(req['input_datarate'])

                # Requested chain of VM and VNF instances
                self.chain = deepcopy(req['chain'])

                self.prefix = ''

                self.optords = None

                # The set of permutations that should be used for the optional
                # orders in the request, if any
                #self.forceOrder = dict()

        def __str__(self):
                return "REQUEST:\n" + "UF: " + str(self.UF) + "\n" + "r: " + str(self.r) + "\n" + "A: " + str(self.A) + "\n" + "l_req: " + str(self.l_req) + "\n" + "Input datarate: " + str(self.input_datarate) + "\n" + "Chain: " + str(self.chain)

        def add_prefix(self, i):
                self.prefix = "req" + str(i)
                UF_tmp = {}
                for uf in list(self.UF.keys()):
                        UF_tmp["req" + str(i) + "_" + uf] = self.UF.pop(uf)
                self.UF = UF_tmp
                for x,u in enumerate(self.U):
                        self.U[x] = "req" + str(i) + "_" + u
                r_tmp = {}
                for r in list(self.r.keys()):
                        r_tmp["req" + str(i) + "_" + r] = self.r.pop(r)
                self.r = r_tmp
                A_tmp = {}
                for a in list(self.A.keys()):
                        A_tmp["req" + str(i) + "_" + a] = self.A.pop(a)
                self.A = A_tmp
                l_req_tmp = {}
                for (x,y) in list(self.l_req.keys()):
                        l_req_tmp[("req" + str(i) + "_" + x, "req" + str(i) + "_" + y)] = self.l_req.pop((x,y))
                self.l_req = l_req_tmp
                self.chain = self.chain.replace("u", "req" + str(i) + "_u")
                self.chain = self.chain.replace("a", "req" + str(i) + "_a")
