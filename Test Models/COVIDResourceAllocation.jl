module COVIDResourceAllocation

include("../models/PatientAllocation.jl")
include("../models/ReusableResourceAllocation.jl")
include("../models/DisposableResourceAllocation.jl")
include("../models/PatientNurseAllocation.jl")
include("../models/NurseAllocation.jl")
# include("../models/PatientNurseAllocationNew.jl")
include("../models/PatientNurseAllocationNew1.jl")

include("../processing/BedsData.jl")
include("../processing/ForecastData.jl")
include("../processing/GeographicData.jl")
include("../processing/NurseData.jl")

include("../src/util/PatientAllocationResults.jl")
include("../src/util/NurseAllocationResults.jl")

import .PatientAllocation: patient_redistribution 
import .PatientNurseAllocation: patient_nurse_allocation
import .ReusableResourceAllocation: reusable_resource_allocation
import .NurseAllocation: nurse_allocation
# import .PatientNurseAllocationNew: patient_nurse_allocation_new
import .PatientNurseAllocationNew1: patient_nurse_allocation_new1


import .ForecastData: forecast
import .BedsData: n_beds
import .GeographicData: adjacencies
import .NurseData: n_nurses
import .PatientAllocationResults
import .NurseAllocationResults

export reusable_resource_allocation, patient_redistribution, patient_nurse_allocation,nurse_allocation,patient_nurse_allocation_new1
export forecast, adjacencies, n_nurses, n_beds
export PatientAllocationResults, NurseAllocationResults

end
