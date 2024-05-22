#!/bin/csh

# Define directories: 
set input_dir = "/pscratch/sd/q/qinyi/E3SMv2_init/v2.LR.historical_0151/archive/atm/hist"
set output_dir = "/pscratch/sd/p/plutzner/E3SM/E3SMv2data/member2/monthly_bilinear"
set mapping_file = "/pscratch/sd/p/plma/shared/for_jolan/map_ne30pg2_to_cmip6_180x360_bilin.20230823.nc"


# Load the module for ncremap
source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_pm-cpu.csh

# Create output directory
if (! -d "$output_dir") then
    mkdir -p "$output_dir"
endif

#echo "Listing all .nc files in $input_dir:"
set input_files = `find "$input_dir" -name "v2.LR.historical_0151.eam.h0.*.nc" -type f`

# Check if there are any .nc files to process
if ( $#input_files == 0 ) then
    echo "No .nc files found in $input_dir."
    exit 1
endif

# Iterate over each input file
foreach input_file ($input_files)
    set base_name = `basename "$input_file"`

    # Extract the date part of the filename
    set date_part = `echo "$base_name" | sed -n 's/.*\.h0\.\([0-9]\{4\}-[0-9]\{2\}\)\.nc/\1/p'`

    # Extract the specific part of the filename
    set base_name_short = `echo "$base_name" | sed -n 's/\(v2\.LR\.historical_0151\.eam\.h0\.\).*/\1/p'`

    # Define output file name
    set output_file = "$output_dir/${base_name_short}$date_part.bil.nc"

    # Check if the output file already exists
    if (-e "$output_file") then
        echo "Output file $output_file already exists. Skipping..."
    else
        echo "Output will be saved to: $output_file"

        # Execute ncremap command
        ncremap -i "$input_file" -o "$output_file" -m "$mapping_file"
    endif
end
