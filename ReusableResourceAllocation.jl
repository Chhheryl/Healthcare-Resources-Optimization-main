{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "module ReusableResourceAllocation\n",
    "\n",
    "using JuMP\n",
    "using Gurobi\n",
    "\n",
    "using LinearAlgebra\n",
    "using MathOptInterface\n",
    "\n",
    "export reusable_resource_allocation\n",
    "\n",
    "\n",
    "function reusable_resource_allocation(\n",
    "\t\tinitial_supply::Array{<:Real,1},\n",
    "\t\tsupply::Array{<:Real,2},\n",
    "\t\tdemand::Array{<:Real,2},\n",
    "\t\tadj_matrix::BitArray{2};\n",
    "\t\tobj_dir::Symbol=:shortage,\n",
    "\t\tsend_new_only::Bool=false,\n",
    "\t\tsendrecieve_switch_time::Int=0,\n",
    "\t\tmin_send_amt::Real=0,\n",
    "\t\tsmoothness_penalty::Real=0,\n",
    "\t\tsetup_cost::Real=0,\n",
    "\t\tsent_penalty::Real=0,\n",
    "\t\tverbose::Bool=false,\n",
    ")\n",
    "\tN, T = size(supply)\n",
    "\t@assert(size(initial_supply, 1) == N)\n",
    "\t@assert(size(demand, 1) == N)\n",
    "\t@assert(size(demand, 2) == T)\n",
    "\t@assert(size(adj_matrix, 1) == N)\n",
    "\t@assert(size(adj_matrix, 2) == N)\n",
    "\t@assert(obj_dir in [:shortage, :overflow])\n",
    "\n",
    "\tmodel = Model(Gurobi.Optimizer)\n",
    "\tif !verbose set_silent(model) end\n",
    "\n",
    "\t@variable(model, sent[1:N,1:N,1:T])\n",
    "\t@variable(model, obj_dummy[1:N,1:T] >= 0)\n",
    "\n",
    "\tif min_send_amt <= 0\n",
    "\t\t@constraint(model, sent .>= 0)\n",
    "\telse\n",
    "\t\t@constraint(model, [i=1:N,j=1:N,t=1:T], sent[i,j,t] in MOI.Semicontinuous(Float64(min_send_amt), Inf))\n",
    "\tend\n",
    "\n",
    "\tobjective = @expression(model, sum(obj_dummy))\n",
    "\tif sent_penalty > 0\n",
    "\t\tadd_to_expression!(objective, sent_penalty*sum(sent))\n",
    "\tend\n",
    "\tif smoothness_penalty > 0\n",
    "\t\t@variable(model, smoothness_dummy[i=1:N,j=1:N,t=1:T-1] >= 0)\n",
    "\t\t@constraint(model, [t=1:T-1],  (sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])\n",
    "\t\t@constraint(model, [t=1:T-1], -(sent[:,:,t] - sent[:,:,t+1]) .<= smoothness_dummy[:,:,t])\n",
    "\n",
    "\t\tadd_to_expression!(objective, smoothness_penalty * sum(smoothness_dummy))\n",
    "\t\tadd_to_expression!(objective, smoothness_penalty * sum(sent[:,:,1]))\n",
    "\tend\n",
    "\tif setup_cost > 0\n",
    "\t\t@variable(model, setup_dummy[i=1:N,j=i+1:N], Bin)\n",
    "\t\t@constraint(model, [i=1:N,j=i+1:N], [1-setup_dummy[i,j], sum(sent[i,j,:])+sum(sent[j,i,:])] in MOI.SOS1([1.0, 1.0]))\n",
    "\t\tadd_to_expression!(objective, setup_cost*sum(setup_dummy))\n",
    "\tend\n",
    "\t@objective(model, Min, objective)\n",
    "\n",
    "\tif send_new_only\n",
    "\t\t@constraint(model, [t=1:T],\n",
    "\t\t\tsum(sent[:,:,t], dims=2) .<= max.(0, supply[:,t])\n",
    "\t\t)\n",
    "\telse\n",
    "\t\t@constraint(model, [i=1:N,t=1:T],\n",
    "\t\t\tsum(sent[i,:,t]) <=\n",
    "\t\t\t\tinitial_supply[i]\n",
    "\t\t\t\t+ sum(supply[i,1:t])\n",
    "\t\t\t\t- sum(sent[i,:,1:t-1])\n",
    "\t\t\t\t+ sum(sent[:,i,1:t-1])\n",
    "\t\t)\n",
    "\tend\n",
    "\n",
    "\tfor i = 1:N\n",
    "\t\tfor j = 1:N\n",
    "\t\t\tif ~adj_matrix[i,j]\n",
    "\t\t\t\t@constraint(model, sum(sent[i,j,:]) .== 0)\n",
    "\t\t\tend\n",
    "\t\tend\n",
    "\tend\n",
    "\n",
    "\tif sendrecieve_switch_time > 0\n",
    "\t\t@constraint(model, [i=1:N,t=1:T-1],\n",
    "\t\t\t[sum(sent[:,i,t]), sum(sent[i,:,t:min(t+sendrecieve_switch_time,T)])] in MOI.SOS1([1.0, 1.0])\n",
    "\t\t)\n",
    "\t\t@constraint(model, [i=1:N,t=1:T-1],\n",
    "\t\t\t[sum(sent[:,i,t:min(t+sendrecieve_switch_time,T)]), sum(sent[i,:,t])] in MOI.SOS1([1.0, 1.0])\n",
    "\t\t)\n",
    "\tend\n",
    "\n",
    "\tflip_sign = (obj_dir == :shortage) ? 1 : -1\n",
    "\tz1, z2 = (obj_dir == :shortage) ? (0, -1) : (-1, 0)\n",
    "\t@constraint(model, [i=1:N,t=1:T],\n",
    "\t\tobj_dummy[i,t] >= flip_sign * (\n",
    "\t\t\tdemand[i,t] - (\n",
    "\t\t\t\tinitial_supply[i]\n",
    "\t\t\t\t+ sum(supply[i,1:t])\n",
    "\t\t\t\t- sum(sent[i,:,1:t+z1])\n",
    "\t\t\t\t+ sum(sent[:,i,1:t+z2])\n",
    "\t\t\t)\n",
    "\t\t)\n",
    "\t)\n",
    "\n",
    "\toptimize!(model)\n",
    "\treturn model\n",
    "end\n",
    "\n",
    "end;"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}