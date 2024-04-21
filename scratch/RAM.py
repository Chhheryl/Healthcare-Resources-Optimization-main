import numpy as np 
import pulp

def reusable_resource_allocation(
        initial_supply,
        supply,
        demand,
        adj_matrix,
        obj_dir='shortage',
        send_new_only=False,
        send_receive_switch_time=0,
        min_send_amt=0,
        smoothness_penalty=0,
        setup_cost=0,
        sent_penalty=1,
        verbose=False):
    '''
        description: models a linear optimization problem to manage the allocation of reusable resources across different locations over time
    
        initial_supply: An array representing the initial quantity of resources available at each location at the beginning of the planning horizon.

        supply: A 2D array (matrix) where each element [i][t] represents the additional supply of resources that becomes available at location i at time period t.

        demand: A 2D array (matrix) where each element [i][t] indicates the demand for resources at location i at time period t.

        adj_matrix: A 2D Boolean array (matrix) where each element [i][j] is True if resources can be sent directly from location i to location j.
                    If the value is False, no resources can be transferred between these two locations.

        obj_dir (Objective Direction, default 'shortage'): A string that can either be 'shortage' or 'overflow'. If set to 'shortage', the objective is to 
                                                           minimize the unmet demand (shortage) at each location and time period. If set to 'overflow', the
                                                           objective could be adjusted to minimize any excess of resources (though, in the current setup, 
                                                           we focus only on shortage).

        send_new_only (default False): A Boolean flag indicating whether resources can only be sent from the current period’s supply. 
                                       If True, resources from initial or cumulative supplies cannot be used, restricting the transfers to what is 
                                       freshly supplied in each period.

        send_receive_switch_time (default 0): An integer indicating a delay or switch time between sending and receiving resources. 
                                              This would impose additional constraints to manage the timing of transfers, but it's not 
                                            fully implemented in the provided script.

        min_send_amt (default 0): A non-negative value that sets a minimum threshold for the amount of resources sent in any transfer. 
                                  If set to a value greater than zero, transfers below this amount are not allowed.

        smoothness_penalty (default 0): A non-negative value used as a multiplier to penalize changes in the amount of resources sent between 
                                        consecutive periods. This can be used to enforce a more stable or smooth allocation over time.

        setup_cost (default 0): A non-negative value representing the cost to set up a transfer of resources between locations. 
                                If positive, this adds a fixed cost for each pair of locations between which resources are transferred, 
                                encouraging fewer unique transfer routes.

        
        sent_penalty (default 0): A non-negative value that penalizes the total amount of resources sent. 
                                  This can be used to minimize the total volume of transfers across the network.

        verbose (default False): A Boolean flag that, when set to True, allows the solver's output to be printed to the console, 
                                 providing details on the optimization process.
    '''
    
    import pulp
    import numpy as np

    # get the number of nodes (healthcare centers) in the network: N 
    # get the number of time periods: T
    N, T = supply.shape
    
    # Model: indicates that the objective of the linear programming problem is to minimize the value of the objective function
    model = pulp.LpProblem("ResourceAllocation", pulp.LpMinimize)

    # Variables: define the decision variables for your model 
    
    # edges in the network define that hospital i can send resources to hopital j at time t; can be no less than 0 (non-negativity constraint)
    # constructs a matrix of variables 
    sent = [
            [   
                [pulp.LpVariable(f"sent_{i}_{j}_{t}", lowBound=0) for t in range(T)]
                for j in range(N)
            ]
            for i in range(N)]
    
    # placeholder dummy variables that can be used to add addtional constraints to our model
    obj_dummy = [[pulp.LpVariable(f"obj_dummy_{i}_{t}", lowBound=0)
                  for t in range(T)] for i in range(N)]
    
    ####################################
    # ADD THE CONSTRAINTS TO THE MODEL #
    ####################################
    # Handle minimum send amount
    if min_send_amt > 0:
        for i in range(N):
            for j in range(N):
                for t in range(T):
                    model += sent[i][j][t] >= min_send_amt * (sent[i][j][t] > 0)

    # Constraints for resource transfer
    if send_new_only:
        for t in range(T):
            for i in range(N):
                # for a given time period t you can only send as much resources from node i 
                # as you just recived in time period t 
                model += pulp.lpSum(sent[i][j][t] for j in range(N)) <= max(0, supply[i, t])
    else:
        for t in range(T):
            for i in range(N):
                # resources received from hospitals i 
                received = pulp.lpSum(sent[j][i][u] for j in range(N) for u in range(t+1))
                # resources sent out from hospital i 
                sent_out = pulp.lpSum(sent[i][j][u] for j in range(N) for u in range(t+1))
                model += received - sent_out <= initial_supply[i] + np.sum(supply[i, :t+1])

    # Adjacency matrix constraints
    for i in range(N):
        for j in range(N):
            # if 0; i.e we do not have an edge between i and j 
            if not adj_matrix[i, j]:
                # make sure that the amount sent form i to j for t in range T is 0 
                model += pulp.lpSum(sent[i][j][t] for t in range(T)) == 0

    # Smoothness penalty handling
    # if smoothness_penalty > 0:
    #     for i in range(N):
    #         for j in range(N):
    #             for t in range(1, T):
    #                 diff = pulp.LpVariable(f"diff_{i}_{j}_{t}", lowBound=0)
    #                 model += diff >= sent[i][j][t] - sent[i][j][t-1]
    #                 model += diff >= sent[i][j][t-1] - sent[i][j][t]
    #                 model += smoothness_penalty * diff

    # Objective Function
    # modify obj_dummy if we want to add additional constraints
    model += pulp.lpSum([obj_dummy[i][t] for i in range(N) for t in range(T)]) \
             + sent_penalty * pulp.lpSum([sent[i][j][t] for i in range(N) for j in range(N) for t in range(T)])

    # Solve the model
    if verbose:
        model.solve()
    else:
        model.solve(pulp.PULP_CBC_CMD(msg=False))

    if model.status == pulp.LpStatusOptimal:
        print("Optimal solution found!")
        return { "status": "Optimal", "objective": pulp.value(model.objective) }
    else:
        print("Problem has no optimal solution.")
        return { "status": "Not solved", "objective": None }

# # Example test case
# initial_supply = np.array([10, 15, 20])
# supply = np.array([[5, 10], [10, 15], [5, 10]])
# demand = np.array([[8, 6], [25, 15], [10, 10]])
# adj_matrix = np.array([[True, True, False], [True, True, True], [False, True, True]])

# result = reusable_resource_allocation(initial_supply, supply, demand, adj_matrix, verbose=True)
# print(result)
