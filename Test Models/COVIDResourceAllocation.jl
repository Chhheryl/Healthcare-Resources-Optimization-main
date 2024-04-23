module COVIDResourceAllocation

include("../models/PatientAllocation.jl")
include("../models/ReusableResourceAllocation.jl")
include("../models/DisposableResourceAllocation.jl")
include("../models/PatientNurseAllocation.jl")
include("../models/PatientNurseAllocationNew.jl")

include("../processing/BedsData.jl")
include("../processing/ForecastData.jl")
include("../processing/GeographicData.jl")
include("../processing/NurseData.jl")

include("../src/util/PatientAllocationResults.jl")
include("../src/util/NurseAllocationResults.jl")

import .PatientAllocation: patient_redistribution 
import .PatientNurseAllocation: patient_nurse_allocation
import .ReusableResourceAllocation: reusable_resource_allocation
import .PatientNurseAllocationNew: patient_nurse_allocation_new

import .ForecastData: forecast
import .BedsData: n_beds
import .GeographicData: adjacencies
import .NurseData: n_nurses
import .PatientAllocationResults
import .NurseAllocationResults

export reusable_resource_allocation, patient_redistribution, patient_nurse_allocation, patient_nurse_allocation_new
export forecast, adjacencies, n_nurses, n_beds
export PatientAllocationResults, NurseAllocationResults

end
