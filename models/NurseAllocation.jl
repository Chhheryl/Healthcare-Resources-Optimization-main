module NurseAllocation

using JuMP
using Gurobi
using LinearAlgebra
using MathOptInterface

export nurse_allocation


function nurse_allocation(
		initial_nurses::Array{<:Real,1},
		demand::Array{<:Real,2},
		adj_matrix::BitArray{2},
		isolation_spot::Array{<:Real,1};
		sent_penalty::Real=0,
		smoothness_penalty::Real=0,

		no_artificial_shortage::Bool=false,
		no_worse_shortage::Bool=false,
		fully_connected::Bool=false,

		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		setup_cost::Real=0,

		verbose::Bool=false,
)
	#=
	initial_nurses (Array{<Real,1}): 
	An array representing the initial number of nurses available at each location or unit.
	This is the starting point for the allocation process.

	demand (Array{<Real,2}): 
	A two-dimensional array indicating the number of nurses required at each location for each time period.
	Each row corresponds to a location, and each column corresponds to a time period.

	adj_matrix (BitArray{2}): 
	A square binary matrix (adjacency matrix) that defines the connectivity between locations.
	If adj_matrix[i, j] is true, nurses can be sent from location i to location j. 
	This parameter is critical for defining which transfers are allowed.

	------------------------------------------------------------------------------------------------------------
	
	sent_penalty (Real): 
	A penalty added to the objective function for sending nurses between locations. 
	This parameter discourages excessive transfers to minimize operational disruption and costs.

	smoothness_penalty (Real): 
	A penalty for large variations in the number of nurses sent between periods. 
	This parameter aims to ensure a more consistent flow of nurse allocations over time, promoting stability in 
	staffing levels.

	no_artificial_shortage (Bool): 
	A flag that, when set to true, ensures that no artificial nurse shortages are created due to nurse 
	reallocation if the initial number of nurses at a location is already sufficient to meet the demand.

	no_worse_shortage (Bool): 
	When set, this flag ensures that the nurse shortage does not worsen due to redistribution.
	If a location starts with fewer nurses than needed, the allocation won't allow reducing that 
	number further through transfers.

	fully_connected (Bool):
	If set to true, this ignores the adjacency matrix and allows nurses to be sent between any 
	two locations, treating the network as fully connected.

	sendreceive_switch_time (Int): 
	Defines a time window in which switching between sending and 
	receiving nurses at a location is restricted. This can be used to model operational delays or 
	synchronization requirements.

	min_send_amt (Real): 
	The minimum amount of nurses that must be sent if any are sent at all, ensuring that nurse transfers are not 
	trivially small and thus operationally insignificant.

	setup_cost (Real): 
	A cost incurred for establishing a new transfer route between two locations. This could represent logistical, 
	administrative, or other initial costs associated with starting a nurse transfer.

	verbose (Bool):
	Controls the output of solver and model-building information. If true, 
	the model will display more detailed information during the optimization process.
=#

	###############
	#### Setup ####
	###############

	N, T = size(demand)
	@assert(size(initial_nurses) == (N,))
	@assert(size(adj_matrix) == (N,N))

	###############
	#### Model ####
	###############

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	###############
	## Variables ##
	###############

	@variable(model, sent[1:N,1:N,1:T] >= 0)
	@variable(model, obj_dummy[1:N,1:T] >= 0)

	#################
	## Expressions ##
	#################

	# compute active nurses
	@expression(model, active_nurses[i=1:N,t=0:T],
		initial_nurses[i]
		- sum(sent[i,:,1:t])
		+ sum(sent[:,i,1:t])
	)

	objective = @expression(model, sum(obj_dummy))

	######################
	## Hard Constraints ##
	######################

	# sent nurses ≦ active nurses
	@constraint(model, [i=1:N,t=1:T], sum(sent[i,:,t]) <= active_nurses[i,t-1])

	# objective
	@constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= demand[i,t] - active_nurses[i,t])

	for i in 1:N, t in 1:T
		fix(sent[i,i,t], 0, force=true)
	end
	if !fully_connected
		for i in 1:N, j in 1:N
			if ~adj_matrix[i,j]
				for t in 1:T
					fix(sent[i,j,t], 0, force=true)
				end
			end
		end
	end

	##########################
	## Optional Constraints ##
	##########################

	if no_artificial_shortage
		for i in 1:N, t in 1:T
			if initial_nurses[i] >= demand[i,t]
				@constraint(model, active_nurses[i,t] >= demand[i,t])
			end
		end
	end

	if no_worse_shortage
		for i in 1:N, t in 1:T
			if initial_nurses[i] < demand[i,t]
				@constraint(model, active_nurses[i,t] >= initial_nurses[i])
			end
		end
	end

	# active nurses ≧ 1/2 initial nurses
	@constraint(model, [i=1:N,t=1:T], active_nurses[i,t] >= 0.5 * initial_nurses[i])

	if min_send_amt <= 0
		@constraint(model, sent .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0])
		)
		@constraint(model, [i=1:N,t=1:T-1],
			[sum(sent[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])
		)
	end

	########################
	## Optional Penalties ##
	########################

	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*sum(sent))
	end

	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))
	end

	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))
	end

	if length(isolation_spot) > 0 
		isolated_nodes = findall(isolation_spot .== 1)
		non_isolated_nodes = findall(isolation_spot .== 0)
		for i in isolated_nodes
			# Add terms for sent nurses from isolated nodes to nonisolated nodes
			add_to_expression!(objective, sum(1000 * sent[i, j, t] for j in non_isolated_nodes, t in 1:T))
			#severity = [obj_dummy[i,t] > 0 ? 1.0 : 100.0 for i in 1:N]
			add_to_expression!(objective, sum(obj_dummy[i,t] for t in 1:T))
		end
	end

	###############
	#### Solve ####
	###############

	@objective(model, Min, objective)
	optimize!(model)

	return model
end

end;