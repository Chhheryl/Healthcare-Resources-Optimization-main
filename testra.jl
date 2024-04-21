using JuMP
using Gurobi

# Assuming your model code is correctly placed and named as per your earlier description
include("/Users/joshuagrajales/Desktop/Healthcare-Resources-Optimization-main/ra.jl")  # Adjust the path as needed

function test_resource_allocation()
    # Node setup: 3 nodes, 2 time periods
    N = 3
    T = 2

    # Initial supplies at each node
    initial_supply = [10, 5, 30]  # Node 3 has a surplus

    # Supplies over time (none additional for simplicity)
    supply = zeros(Int, N, T)

    # Demands at each node, across time
    demand = [
        15 20;  # Node 1 needs more resources
        10 15;  # Node 2 needs more resources
        25 25   # Node 3 needs exactly what it has initially and over time
    ]

    # Adjacency matrix (all nodes can send to each other)
    adj_matrix = trues(N, N)

    # Additional options to ensure there is resource movement
    options = (
        obj_dir = :shortage,
        send_new_only = false,
        sendrecieve_switch_time = 0,
        min_send_amt = 1,  # Ensures that any non-zero `sent` is at least 1
        smoothness_penalty = 0,
        setup_cost = 0,
        sent_penalty = 0,
        verbose = true
    )

    # Create and solve the model
    model = ReusableResourceAllocation.reusable_resource_allocation(
        initial_supply,
        supply,
        demand,
        adj_matrix;
        options...
    )

    # Fetch and display results
    sent_values = JuMP.value.(model[:sent])
    println("Sent values between nodes over time:")
    display(sent_values)

    return model, sent_values
end

# Execute the test function
model, sent_values = test_resource_allocation()




