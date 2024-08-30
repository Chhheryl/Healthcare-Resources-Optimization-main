module PatientNurseAllocationNew1

using JuMP
using Gurobi

using LinearAlgebra
using Distributions
using MathOptInterface

export patient_nurse_allocation_new


function patient_nurse_allocation_new1(
		initial_patients::Array{<:Real,1},
		discharged_patients::Array{<:Real,2},
		admitted_patients::Array{<:Real,2},
		initial_nurses::Array{<:Real,1},
		adj_matrix::BitArray{2},
		isolation_spot::Array{<:Real,1};
		los=11,
		nurse_days_per_patient_day::Real=2.0,
		sendreceive_switch_time::Int=0,
		min_send_amt::Real=0,
		smoothness_penalty::Real=0,
		setup_cost::Real=0,
		sent_penalty::Real=0,
		balancing_thresh_patients::Real=1.0,
		balancing_penalty_patients::Real=0,
		nurse_target_load::Real=1.25,
		nurse_target_load_gap::Real=0.25,
		nurse_load_penalty::Real=0,
		disallow_nurse_shortage_sent::Bool=false,
		disallow_nurse_shortage_newpatients::Bool=false,
		severity_weighting::Bool=false,
		no_artificial_overflow::Bool=false,
		no_artificial_shortage::Bool=false,
		verbose::Bool=false,
)
	N, T = size(admitted_patients)
	@assert(size(initial_patients, 1) == N)
	@assert(size(initial_nurses, 1) == N)
	@assert(size(adj_matrix) == (N,N))
	@assert(size(discharged_patients) == (N, T))

	# L will be an array containing 'los' ones followed by 'T-los' zeros
	# each value is the probablity of one still staying at hospital at a given time
	L = nothing
	if isa(los, Int)
		L = vcat(ones(Int, los), zeros(Int, T-los))
	elseif isa(los, Array{<:Real,1})
		if length(los) >= T
			L = los
		else
			L = vcat(los, zeros(Float64, T-length(los)))
		end
	elseif isa(los, Distribution)
		L = 1.0 .- cdf.(los, 0:T)
	else
		error("Invalid length of stay distribution")
	end

	model = Model(Gurobi.Optimizer)
	if !verbose set_silent(model) end

	# patients sent & nurses sent 
	@variable(model, sentpatients[1:N,1:N,1:T] >= 0)
	@variable(model, sentnurses[1:N,1:N,1:T] >= 0)


	@variable(model, obj_dummy[1:N,1:T] >= 0)
	#@variable(model, obj_dummy_nurses[1:N,1:T] >= 0)

	# enforce minimum transfer amount if enabled
	if min_send_amt <= 0
		@constraint(model, sentpatients .>= 0)
		@constraint(model, sentnurses .>= 0)
		#@constraint(model,sentdisresource .>= 0)
		#@constraint(model,sentreusableresource .>= 0)
	else
		@constraint(model, [i=1:N,j=1:N,t=1:T], sentpatients[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
		@constraint(model, [i=1:N,j=1:N,t=1:T], sentnurses[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
		#@constraint(model, [i=1:N,j=1:N,t=1:T], sentdisresource[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
		#@constraint(model, [i=1:N,j=1:N,t=1:T], sentreusableresource[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))
	end

	objective = @expression(model, sum(obj_dummy))

	active_patients_null = [(
			initial_patients[i]
			- discharged_patients[i,t]
			+ sum(L[t-t₁+1] * admitted_patients[i,t₁] for t₁ in 1:t)
		) for i in 1:N, t in 1:T
	]

	# compute active nurses
	@expression(model, active_nurses[i=1:N,t=0:T],
		initial_nurses[i]
		- sum(sentnurses[i,:,1:t])
		+ sum(sentnurses[:,i,1:t])
	)

	# ensure the number of active nurses is non-negative
	@constraint(model, [i=1:N,t=1:T], active_nurses[i,t] >= 0)


	# measure the severity weights for each ward. The less the weight, the greater the severity.
	# If enabled, the contribution of 'obj_dummy_patients' is weighted based on the severity of ward overloads
	if severity_weighting
		max_load_null = [maximum(active_patients_null[i,t]*nurse_days_per_patient_day / active_nurses[i,t-1]) for i in 1:N, t in 1:T]
		# if the maximum load per bed is greater than 1, set the severity weight to 1
		# else, set the severity weight to 10 (not severe so don't transfer it)
		severity_weight = [max_load_null[i] > 1 ? 1.0 : 10.0 for i in 1:N]
		
		add_to_expression!(objective, dot(sum(obj_dummy, dims=2), severity_weight))
	else
		add_to_expression!(objective, sum(obj_dummy))
	end

	# penalize total sent if enabled
	if sent_penalty > 0
		add_to_expression!(objective, sent_penalty*2*sum(sentpatients))
		add_to_expression!(objective, sent_penalty*sum(sentnurses))
		#add_to_expression!(objective, sent_penalty*sum(sentdisresource))
		#add_to_expression!(objective, sent_penalty*sum(sentreusableresource))
	end

	# penalize non-smoothness in sent patients if enabled
	# this is to ensure that it's feasible to plan ahead, fixing the setup amounts 
	if smoothness_penalty > 0
		@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)
		@constraint(model, [t=1:T-1],  (sentpatients[:,:,t] - sentpatients[:,:,t+1]) .<= smoothness_dummy[:,:,t])
		@constraint(model, [t=1:T-1], -(sentpatients[:,:,t] - sentpatients[:,:,t+1]) .<= smoothness_dummy[:,:,t])

		add_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))
		add_to_expression!(objective, smoothness_penalty * sum(sentpatients[:,:,1]))
	end

	# add setup costs if enabled
	if setup_cost > 0
		@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)
		@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sentpatients[i,j,:])+sum(sentpatients[j,i,:]) + sum(sentnurses[i,j,:])+sum(sentnurses[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		add_to_expression!(objective, setup_cost*sum(setup_dummy))

		# @variable(model, setup_dummy_nurses[i=1:N,j=i+1:N], Bin)
		# @constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy_nurses[i,j], sum(sentnurses[i,j,:])+sum(sentnurses[j,i,:])] in MOI.SOS1([1.0, 1.0]))
		# add_to_expression!(objective, setup_cost*sum(setup_dummy_nurses))
	end

	# only send patients between connected locations
	for i = 1:N
		@constraint(model, sentnurses[i,i,:] .== 0)
		@constraint(model, sentpatients[i,i,:] .== 0)
		for j = 1:N
			if ~adj_matrix[i,j]
				@constraint(model, sentpatients[i,j,:] .== 0)
				@constraint(model, sentnurses[i,j,:] .== 0)
			end
		end
	end

	# enforce a minimum time between sending and receiving
	if sendreceive_switch_time > 0
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentpatients[:,i,t]), sum(sentpatients[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentpatients[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sentpatients[i,:,t])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentnurses[:,i,t]), sum(sentnurses[i,:,t:min(t+sendreceive_switch_time,T)])] in MOI.SOS1([1.0, 1.0]))
		@constraint(model, [i=1:N,t=1:T-1], [sum(sentnurses[:,i,t:min(t+sendreceive_switch_time,T)]), sum(sentnurses[i,:,t])] in MOI.SOS1([1.0, 1.0]))
	end

	# send new patients only
	@constraint(model, [t=1:T], sum(sentpatients[:,:,t], dims=2) .<= admitted_patients[:,t])


	# expression for the number of active patients
	@expression(model, active_patients[i=1:N,t=1:T],
		initial_patients[i]
		- discharged_patients[i,t]
		+ sum(L[t-t₁+1] * (
			admitted_patients[i,t₁]
			- sum(sentpatients[i,:,t₁])
			+ sum(sentpatients[:,i,t₁])
		) for t₁ in 1:t)
		+ sum(sentpatients[i,:,t])
	)

	# ensure the number of active patients is non-negative
	#@constraint(model, [i=1:N,t=1:T], active_patients[i,t] >= 0)


	if no_artificial_overflow
		for i in 1:N, t in 1:T
			if active_patients_null[i,t] < active_nurses[i]*(1/nurse_days_per_patient_day)
				@constraint(model, active_patients[i,t] <= active_nurses[i,t-1]*(1/nurse_days_per_patient_day))
			end
		end
	end

	# load balancing for patients
	if balancing_penalty_patients > 0
		@variable(model, balancing_dummy_patients[1:N,1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], balancing_dummy_patients[i,t] >= (active_patients[i,t]*nurse_days_per_patient_day / active_nurses[i,t]) - balancing_thresh_patients)
		add_to_expression!(objective, balancing_penalty_patients * sum(balancing_dummy_patients))
	end

	# objective 
	# @expression(model, patient_overflow[i=1:N,t=1:T], active_patients[i,t] - active_nurses[i,t]*(1/nurse_days_per_patient_day))
	# @constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= patient_overflow[i,t])

	# compute nurse demand
	@expression(model, nurse_demand[i=1:N,t=1:T], active_patients[i,t] * nurse_days_per_patient_day)

	# sent nurses ≦ active nurses
	@constraint(model, [i=1:N,t=1:T], sum(sentnurses[i,:,t]) <= active_nurses[i,t-1])

	# active nurses ≧ 1/2 initial nurses
	@constraint(model, [i=1:N,t=1:T], active_nurses[i,t] >= 0.5 * initial_nurses[i])

	# nurses objective
	@constraint(model, [i=1:N,t=1:T], obj_dummy[i,t] >= nurse_demand[i,t] - active_nurses[i,t])

	nurse_demand_null = active_patients_null .* nurse_days_per_patient_day
	if no_artificial_shortage
		for i in 1:N, t in 1:T
			if nurse_demand_null[i,t] > initial_nurses[i]
				@constraint(model, active_nurses[i,t] >= initial_nurses[i])
			end
			if nurse_demand_null[i,t] <= initial_nurses[i]
				# @constraint(model, active_nurses[i,t] >= nurse_demand_null[i])
				@constraint(model, active_nurses[i,t] >= nurse_demand[i])
			end
		end
	end

	if disallow_nurse_shortage_sent
		# m = 1e-5
		# @variable(model, has_nurse_shortage[i=1:N,t=1:T], Bin)
		# @constraint(model, [i=1:N,t=1:T],     m*(nurse_demand[i,t] - active_nurses[i,t]) <= has_nurse_shortage[i,t])
		# @constraint(model, [i=1:N,t=1:T], 1 + m*(nurse_demand[i,t] - active_nurses[i,t]) >= has_nurse_shortage[i,t])
		# @constraint(model, [i=1:N,t=1:T], has_nurse_shortage[i,t] => {active_nurses[i,t] >= initial_nurses[i]})
		nurse_demand_null = active_patients_null .* nurse_days_per_patient_day
		for i in 1:N, t in 1:T
			if nurse_demand_null[i,t] >= initial_nurses[i]
				@constraint(model, sum(sentnurses[:,i,1:t]) >= sum(sentnurses[i,:,1:t]))
			end
		end
	end

	if disallow_nurse_shortage_newpatients
		m = 1e-5
		ts(t) = max(1,t-round(Int,mean(los))+1)
		@variable(model, has_outside_patients[i=1:N,t=1:T], Bin)
		@constraint(model, [i=1:N,t=1:T],     m*(sum(sentpatients[:,i,ts(t):t]) - sum(sentpatients[i,:,ts(t):t])) <= has_outside_patients[i,t])
		@constraint(model, [i=1:N,t=1:T], 1 + m*(sum(sentpatients[:,i,ts(t):t]) - sum(sentpatients[i,:,ts(t):t])) >= has_outside_patients[i,t])
		@constraint(model, [i=1:N,t=1:T], has_outside_patients[i,t] => {active_nurses[i,t] >= nurse_demand[i,t]})
	end

	# nurse load
	if nurse_load_penalty > 0
		@variable(model, load_dummy_nurses_abs[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T],  (active_nurses[i,t] - nurse_target_load*nurse_demand[i,t]) <= load_dummy_nurses_abs[i,t])
		@constraint(model, [i=1:N,t=1:T], -(active_nurses[i,t] - nurse_target_load*nurse_demand[i,t]) <= load_dummy_nurses_abs[i,t])

		@variable(model, load_dummy_nurses[i=1:N,t=1:T] >= 0)
		@constraint(model, [i=1:N,t=1:T], load_dummy_nurses[i,t] >= load_dummy_nurses_abs[i,t] - nurse_target_load_gap*nurse_demand[i,t])
		add_to_expression!(objective, nurse_load_penalty * sum(load_dummy_nurses))
	end

	if length(isolation_spot) > 0 
		isolated_nodes = findall(isolation_spot .== 1)
		non_isolated_nodes = findall(isolation_spot .== 0)
		for i in isolated_nodes
			# Add terms for sent patients from noisolated nodes to isolation spot i
			add_to_expression!(objective, sum(1000 * sentpatients[j, i, t] for j in non_isolated_nodes, t in 1:T))
			# Add terms for sent patients from isolation spot i to  noisolated nodes
			add_to_expression!(objective, sum(1000 * sentpatients[i, j, t] for j in non_isolated_nodes, t in 1:T))
			# Add terms for sent nurses from isolated nodes to nonisolated nodes
			add_to_expression!(objective, sum(1000 * sentnurses[i, j, t] for j in non_isolated_nodes, t in 1:T))
			#severity = [obj_dummy[i,t] > 0 ? 1.0 : 100.0 for i in 1:N]
			add_to_expression!(objective, sum(obj_dummy[i,t] for t in 1:T))
		end
	end
	

	@objective(model, Min, objective)

	optimize!(model)
	return model
end
end;