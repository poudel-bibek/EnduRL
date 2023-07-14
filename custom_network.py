"""
A different kind of a bottleneck
"""

# import Flow's base network class
from flow.networks import Network
import numpy as np
# Additional params are defined here
# They are accessed as net_params.additional_params["param"]
# Make sure to copy and pass the same net params when this network is used
ADDITIONAL_NET_PARAMS = {
    "speed_limit": 20,
}

# define the network class, and inherit properties from the base network class
class CustomBottleneckNetwork(Network):
    """
    Nodes, edges, routes

    """
    
    def specify_nodes(self, net_params):
        """
        Positions of select few point in the network
        id: name,
        x, y: coordinates 
        Nodes dont need to consider lanes
        """
        nodes = [
            # Left: Start
            {"id": "left0", "x": 0, "y": 0},
            {"id": "left1", "x": 0, "y": 15},
            # Middle: 
            {"id": "middle0", "x": 200, "y": 0},
            {"id": "middle1", "x": 400, "y": 0},

            # Right: End
            {"id": "right0", "x": 600, "y": 0},
            {"id": "right1", "x": 600, "y": 10},
        ]

        return nodes
    
    def specify_edges(self, net_params):
        """
        Nodes are connected by edges
        id: name,
        from: start node,
        to: end node,
        length: length of the edge,
        numLanes: number of lanes,
        speed: speed limit,
        other sumo related attributes: https://sumo.dlr.de/docs/Networks/PlainXML.html#Edge_Descriptions
        One useful attribute is "shape" which is a list of coordinates (a series of sub-nodes) of the edge
        """
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0", 
                "from": "left0", 
                "to": "middle0", 
                "length": 500, 
                "numLanes": 4, # get from params
                "speed": speed_limit,},
            {
                "id": "edge1",
                "from": "left1",
                "to": "middle0",
                "length": 500,
                "numLanes": 2, # get from params
                "speed": speed_limit,},

            {
                "id": "edge2",
                "from": "middle0",
                "to": "middle1",
                "length": 500,
                "numLanes": 4, # get from params
                "speed": speed_limit,},
            {
                "id": "edge3",
                "from": "middle1",
                "to": "right0",
                "length": 500,
                "numLanes": 3, # get from params
                "speed": speed_limit,},
            {
                "id": "edge4",
                "from": "middle1",
                "to": "right1",
                "length": 500,
                "numLanes": 2, # get from params
                "speed": speed_limit,},

        ]

        return edges

    
    def specify_routes(self, net_params):
        """
        Routes: sequence of edges the vehicles can traverse before restarting
        There can either be a single route per egde or multiple routes per edge
        Additionally, there can also be per vehicle route
        
        Single routes: deterministic e.g., ring road
        Multiple routes: Shocastic, tupes with probability (The sum must be 1)
        """
        routes = {
            # key = edge a vehicle is on, value = list of edges the vehicle can traverse
            "edge0": ["edge0", "edge2", "edge3"],
            "edge1": ["edge1", "edge2", "edge4"],
            "edge2": ["edge3", "edge4"],
            "edge3": ["edge3" ],
            "edge4": ["edge4"],
        }

        return routes

    def specify_edge_starts(self):
        """
        Starting position of vehicles on the edge
        """
        edge_starts = [
            ("edge0", 0),
            ("edge1", 0),
            ("edge2", 500),
            ("edge3", 1000),
            ("edge4", 1000),
        ]

        return edge_starts

    # def specify_connections(self, net_params):
    #     """
    #     Connections: Specify how lanes are connected to each other
    #     """
    #     conn_dic = {}
    #     conn = []
        
    #     # Connect lanes from edge0 to edge2
    #     for i in range(4):
    #         conn += [{
    #             "from": "edge0",
    #             "to": "edge2",
    #             "fromLane": i,
    #             "toLane": int(np.floor(i / 2))
    #         }]
    #     conn_dic["edge2"] = conn
        
    #     conn = []
        
    #     # Connect lanes from edge1 to edge2
    #     for i in range(2):
    #         conn += [{
    #             "from": "edge1",
    #             "to": "edge2",
    #             "fromLane": i,
    #             "toLane": int(np.floor(i / 2)) + 2  # start from the 3rd lane
    #         }]
    #     conn_dic["edge2"] = conn + conn_dic["edge2"]
        
    #     conn = []
        
    #     # Connect lanes from edge2 to edge3
    #     for i in range(4):
    #         conn += [{
    #             "from": "edge2",
    #             "to": "edge3",
    #             "fromLane": i,
    #             "toLane": int(np.floor(i / 3))
    #         }]
    #     conn_dic["edge3"] = conn
        
    #     conn = []
        
    #     # Connect lanes from edge2 to edge4
    #     for i in range(2):
    #         conn += [{
    #             "from": "edge2",
    #             "to": "edge4",
    #             "fromLane": i + 2,  # start from the 3rd lane
    #             "toLane": i
    #         }]
    #     conn_dic["edge4"] = conn

    #     return conn_dic

    def specify_connections(self, net_params):
        """
        Connections: Specify how lanes are connected to each other
        """
        conn_dic = {}
        conn = []
        
        # Connect lanes from edge0 to edge2
        for i in range(4):
            conn += [{
                "from": "edge0",
                "to": "edge2",
                "fromLane": i,
                "toLane": i  # change here
            }]
        conn_dic["edge2"] = conn
        
        conn = []
        
        # Connect lanes from edge1 to edge2
        for i in range(2):
            conn += [{
                "from": "edge1",
                "to": "edge2",
                "fromLane": i,
                "toLane": i + 2  # change here
            }]
        conn_dic["edge2"] = conn + conn_dic["edge2"]
            
        conn = []
            
        # Connect lanes from edge2 to edge3
        for i in range(3):
            conn += [{
                "from": "edge2",
                "to": "edge3",
                "fromLane": i,
                "toLane": i
            }]
        conn_dic["edge3"] = conn
        
        conn = []
            
        # Connect lanes from edge2 to edge4
        for i in range(2):
            conn += [{
                "from": "edge2",
                "to": "edge4",
                "fromLane": i + 2,  # start from the 3rd lane
                "toLane": i
            }]
        conn_dic["edge4"] = conn

        return conn_dic

### FROM EXAMPLES ###
from numpy import pi, sin, cos, linspace


class myNetwork(Network): 

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["radius"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "bottom", "x": 0,  "y": -r},
                 {"id": "right",  "x": r,  "y": 0},
                 {"id": "top",    "x": 0,  "y": r},
                 {"id": "left",   "x": -r, "y": 0}]

        return nodes
    
    def specify_edges(self, net_params):
        r = net_params.additional_params["radius"]
        edgelen = r * pi / 2
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,     
                "from": "bottom", 
                "to": "right", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "right",
                "to": "top",
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "top",
                "to": "left", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]},
            {
                "id": "edge3", 
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "left", 
                "to": "bottom", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }
        ]

        return edges
        
    def specify_routes(self, net_params):
        rts = {"edge0": ["edge0", "edge1", "edge2", "edge3"],
               "edge1": ["edge1", "edge2", "edge3", "edge0"],
               "edge2": ["edge2", "edge3", "edge0", "edge1"],
               "edge3": ["edge3", "edge0", "edge1", "edge2"]}

        return rts
    
    def specify_edge_starts(self):
        r = self.net_params.additional_params["radius"]

        edgestarts = [("edge0", 0),
                      ("edge1", r * 1/2 * pi),
                      ("edge2", r * pi),
                      ("edge3", r * 3/2 * pi)]

        return edgestarts