#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --job-name=dask-root
#SBATCH --mem=30GB
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH --output=job.%j.log
#SBATCH --error=job.%j.err


# start timer and log run info
start=`date +%s`
datelog=`date`

################################### HANDLING ARGS FOR PYTHON #######################################
####################################################################################################
if [[ $# -eq 0 ]] ; then
    echo -e "\n############################## DAXG: Dask-XGBoost on HPC clusters ##############################"
    echo 'Cross-validate, refit or predict a dask-xgboost model on SLURM HPC clusters'
    echo -e "author: matthew bracher-smith (smithmr5@cardiff.ac.uk)\n"
    echo 'NOTE: to run the script with a queue or job specified, you need to pass these to the sbatch'
    echo 'command before the script name. This is true with other additional flags that are slurm args, e.g.'
    echo -e "    sbatch --account myprojectnum -p myqueue --mail-user=email submit_daskxgb_distributed_refit.sh\n"
    echo 'NOTE: to pass args to the python scripts, they must be added after the submission script'
    echo 'These extra arguments have to be given as --flag=option'
    echo 'A full call might include both slurm args and run args, e.g.:'
    echo -e "    sbatch -p myqueue submit_daskxgb_distributed_refit.sh --train=tmp.hdf5\n"
    echo 'options'
    echo '    --train             Full path of hdf5 file for train set'
    echo '    --test              Full path of hdf5 file for test set'
    echo '    --prefix            Prefix pre-prended to all output files. Used to add run info to file names'
    echo '    --basedir           Base directory for dask tmp files. Usually your scratch/workdir space'
    echo '    --cv                Whether to run cross-validation'
    echo '    --iter              Number of iterations of random search'
    echo '    --refit             Whether to run a refit. Requires manual hyper-parameters or --cv to be True'
    echo '    --predict           Whether to predict. Requires --refit to be True or --bst to be supplied'
    echo '    --threads           Number of threads per worker'
    echo '    --time              Time allocated to each worker. Must be less than SBATCH --time given above'
    echo '    --mem               Memory allocated to each worker'
    echo '    --workers           Number of workers in cluster'
    echo '    --bst               Full path for a pre-trained model to use for prediction'
    echo '    --bstcols           Full path for file given columns used to train model given to --bst'
    echo '    --hyperparams       CSV file containing columns ["AUC", "eta", "subsample", "colsample_bytree", "max_depth"]'
    echo '                        Rows are assumed to be average AUC across CV for search and will be sorted by AUC. Note'
    echo '                        that any subsample value given here will overwrite the default value. Only used for'
    echo '                        --refit=True'
    echo '    --out               Full path of directory to write output files to'
    echo '    --cvsubsample       Number of observations to use in CV for HP search. Suggest ~5k'
    echo '    --boosters          Number of boosted trees to learn in CV. Overwritten by --hyperparams for refit.'
    echo '    --hp_round          The round of hyper-parameter search being run. Default is 0. Should be incremented with'
    echo '                        each job submission if HP search is being split across multiple jobs/rounds.'
    echo '    --gpu               Run models on GPUs if True, else CPU.'
    echo '    --interface         Networking interface for connection between workers. Options: ["None", "ib0"]. Ignored'
    echo '                        if local cluster is used instead of slurm cluster. Default is None.'
    echo '    --dask_row_chunks   Number of rows to use in dask when chunking row-wise. Default 100.'
    echo '    --xkey              Key to use for hdf5 files to access X array. Default "x". Options ["x", "x_adjusted"].'
    echo '    --ykey              Key to use for hdf5 files to access y array. Default "y". Options ["y", "y_adjusted"].'
    echo '    --worker_queue      Name of queue to submit worker jobs to. Default None.'
    echo '    --run_shap_inter    Run shap interaction prediction and save if true (affects predict function only)'
    echo -e "################################################################################################\n"
    exit 1
fi

for i in "$@"
do
    case $i in
        --train=*)
            ftrain="${i#*=}"
            ;;
        --test=*)
            ftest="${i#*=}"
            ;;
        --prefix=*)
            prefix="${i#*=}"
            ;;
        --basedir=*)
            basedir="${i#*=}"
            ;;
        --cv=*)
            cv="${i#*=}"
            ;;
        --iter=*)
            iter="${i#*=}"
            ;;
        --refit=*)
            refit="${i#*=}"
            ;;
        --predict=*)
            predict="${i#*=}"
            ;;
        --threads=*)
            threads="${i#*=}"
            ;;
        --time=*)
            time="${i#*=}"
            ;;
        --mem=*)
            mem="${i#*=}"
            ;;
        --workers=*)
            workers="${i#*=}"
            ;;
        --bst=*)
            bst="${i#*=}"
            ;;
        --bstcols=*)
            bstcols="${i#*=}"
            ;;
        --hyperparams=*)
            hyperparams="${i#*=}"
            ;;
        --out=*)
            out="${i#*=}"
            ;;
        --cvsubsample=*)
            cvsubsample="${i#*=}"
            ;;
        --boosters=*)
            boosters="${i#*=}"
            ;;
        --hp_round=*)
            hp_round="${i#*=}"
            ;;
        --gpu=*)
            gpu="${i#*=}"
            ;;
        --interface=*)
            interface="${i#*=}"
            ;;
        --dask_row_chunks=*)
            dask_row_chunks="${i#*=}"
            ;;
        --xkey=*)
            xkey="${i#*=}"
            ;;
        --ykey=*)
            ykey="${i#*=}"
            ;;
        --worker_queue=*)
            worker_queue="${i#*=}"
            ;;
        --run_shap_inter=*)
            run_shap_inter="${i#*=}"
            ;;
        *)
            # unknown option
            ;;
    esac
done

# hardcoded arguments
xgb_model_type='shap'
cv_folds=5

# set default arguments for when args are not passed
iter=${iter:-50}
threads=${threads:-5}
time=${time:-03_50_00}
mem=${mem:-32G}
workers=${workers:-4}
cvsubsample=${cvsubsample:-5000}
boosters=${boosters:-1000}
hyperparams=${hyperparams:-none}
subsample=${subsample:-0.7}
hp_round=${hp_round:-0}
gpu=${gpu:-False}
interface=${interface:-None}
dask_row_chunks=${dask_row_chunks:-100}
xkey=${xkey:-x}
ykey=${ykey:-y}
out=${out}/job_${SLURM_JOB_ID}_${prefix}
worker_queue=${worker_queue:-None}
run_shap_inter=${run_shap_inter:-False}

echo '###################### DAXG: Dask-XGBoost on HPC clusters ######################'
echo 'Cross-validate, refit or predict a dask-xgboost model on SLURM HPC clusters'
echo 'author: matthew bracher-smith (smithmr5@cardiff.ac.uk)'
echo -e "\nrunning with arguments:"
echo "    --train             $ftrain"
echo "    --test              $ftest"
echo "    --prefix            $prefix"
echo "    --basedir           $basedir"
echo "    --server            $server"
echo "    --cv                $cv"
echo "    --iter              $iter"
echo "    --refit             $refit"
echo "    --predict           $predict"
echo "    --threads           $threads"
echo "    --time              $time"
echo "    --mem               $mem"
echo "    --workers           $workers"
echo "    --bst               $bst"
echo "    --bstcols           $bstcols"
echo "    --hyperparams       $hyperparams"
echo "    --out               $out"
echo "    --cvsubsample       $cvsubsample"
echo "    --boosters          $boosters"
echo "    --hp_round          $hp_round"
echo "    --gpu               $gpu"
echo "    --interface         $interface"
echo "    --dask_row_chunks   $dask_row_chunks"
echo "    --xkey              $xkey"
echo "    --ykey              $ykey"
echo "    --worker_queue      $worker_queue"
echo "    --run_shap_inter    $run_shap_inter"
echo -e "\nhardcoded options (edit submission script to change):"
echo "    no. CV folds                    $cv_folds"
echo "    row subsample in xgboost        $subsample (not used)"
echo -e "\nrun date/time: ${datelog}"
echo -e "################################################################################\n"

# Trap command to automatically remove the local temporary folder 120 seconds before job end
trap "rm -r ${basedir}/job_number_${SLURM_JOB_ID}" SIGUSR1
trap "rm -r /tmp/${SLURM_JOB_ID}" SIGUSR1

# checking variables are set for those without defaults (treats empty strings as unset)
if [[ -z $ftrain || -z $ftest || -z $prefix || -z basedir || -z server || -z $cv || -z $refit || -z $predict || -z $out ]]; then
  echo -e "Error: one or more variables listed above are undefined\n"
  exit 1
fi

# handling arg combinations
if [ -s "${bst}" ]; then
    if [ ! -s "${bstcols}" ]; then
        echo "Column file must be supplied to --bstcols if --bst is given"
        exit 1
    fi
fi

if [ "$refit" == "True" ]; then
    if [ "$cv" == "True" ]; then
        echo "Running CV before refit. Any supplied hyperparameters will be ignored."
    else
        if [[ ! -s "${hyperparams}" && ! -d "${hyperparams}" ]]; then
            echo "--hyperparams must be supplied with --refit if --cv=False"
            exit 1
        fi
    fi
fi

if [ "$predict" == "True" ]; then
    if [ ! "$refit" == "True" ]; then
        if [ ! -s "${bst}" ]; then
            echo "Must pass either --refit=True or --bst and --bstcols if --predict=True"
            exit 1
        fi
    fi
fi
echo "Supplied args pass basic logic checks"

################################## SETUP STORAGE AND MODULES #######################################
####################################################################################################

# creating a single out directory where all files from a specific run will be stored together
if [ ! -s "$out" ];then
    mkdir $out
    echo "Created new out dir as ${out}"
else
    echo "Out dir ${out} already exists"
fi

# Create local temporary folder based on job identifier and set permissions
mkdir ${basedir}/job_number_${SLURM_JOB_ID}
chmod 700 ${basedir}/job_number_${SLURM_JOB_ID}

# setup python env
# change to your module/conda env
module load conda/23.11-py311
source activate daxos

# print node name for setting up dask dashboard
echo -e "\n################################################################################"
echo -e "############################# DASK DASHBOARD SETUP #############################"
echo "  Running Dask scheduler on node ${SLURMD_NODENAME}"
echo "  Paste command below into local terminal session to allow tunnelling for the dask dashboard:"
echo "  'ssh -N -L 8787:${SLURMD_NODENAME}:8787 username@server'"
echo "  paste address 'localhost:8787' in local browser session"
echo -e "################################################################################"
echo -e "################################################################################\n"


###################################### Handling file paths #########################################
####################################################################################################
# assuming daxg is kept in the home dir
scriptdir="$HOME/daxg/scripts"

# copying files to tmp dir so jobs can run concurrently without hdf5 threadsafety issues
echo "--> Copying train/test data to new temp location - this may take up to 5-10 minutes..."
cp ${ftrain} ${basedir}/job_number_${SLURM_JOB_ID}
cp ${ftest} ${basedir}/job_number_${SLURM_JOB_ID}

# pulling out filename from path
ftrain_base=$(basename ${ftrain})
ftest_base=$(basename ${ftest})

# creating new full path for tmp data to pass to python
ftrain_tmp="${basedir}/job_number_${SLURM_JOB_ID}/${ftrain_base}"
ftest_tmp="${basedir}/job_number_${SLURM_JOB_ID}/${ftest_base}"
if [ -s "${ftrain_tmp}" ]; then
    echo -e "Temporary train set copied to ${ftrain_tmp}"
else
    echo -e "\nError: temporary train set not created in path: ${ftrain_tmp}"
    exit 1
fi
if [ -s "${ftest_tmp}" ]; then
    echo -e "\nTemporary test set copied to ${ftest_tmp}"
else
    echo -e "\nError: temporary test set not created in path: ${ftest_tmp}"
    exit 1
fi

# remove dask and tmp files after run/error
cleanup_tmp_files () {
    echo -e "\n--> Cleaning up tmp files before exiting\n"
    if [ -s "${basedir}/job_number_${SLURM_JOB_ID}" ]; then
        rm -r ${basedir}/job_number_${SLURM_JOB_ID}
    fi

    if [ -s "/tmp/${SLURM_JOB_ID}" ]; then
        rm -r /tmp/${SLURM_JOB_ID}
    fi
}

# setting output options for xgb models
train_prefix="${prefix}_train"
test_prefix="${prefix}_test"
echo -e "\nAssigning '${train_prefix}' and '${test_prefix}' as prefixes to all train and test files"

################################# RUN XGBOOST CV WITH DASK #########################################
# Runs cross-validation on training data
# NOTE: n_workers_in_cluster * mem_per_worker should cover the size of the dataset in memory several times over
# NOTE: the *SUM* of time_per_worker for both cpu_training and cpu_predict *MUST* be less than the time
# given to SBATCH in the SLURM commands at the top of this script
####################################################################################################

if [ "$cv" == "True" ]; then
    echo -e "\n######################### RUNNING CROSS-VALIDATION #############################\n"
    if [ ! -s "${out}/cv" ];then
        mkdir ${out}/cv
        echo "Created subdirectory ${out}/cv"
    else
        echo "Sub dir ${out}/cv already exists"
    fi

    python3 ${scriptdir}/cv.py \
        --in_ml $ftrain_tmp \
        --n_iter_search $iter \
        --n_boost_round $boosters \
        --n_threads_per_worker $threads \
        --time_per_worker $time \
        --mem_per_worker $mem \
        --n_workers_in_cluster $workers \
        --n_folds $cv_folds \
        --out $out \
        --prefix ${train_prefix}_cvround_${hp_round} \
        --local_dir /tmp/${SLURM_JOB_ID} \
        --cv_subsample_size $cvsubsample \
        --row_chunk_size $dask_row_chunks \
        --cluster distributed \
        --xgb_eval_metric aucpr \
        --verbose True \
        --gpu $gpu \
        --interface $interface \
        --xkey $xkey \
        --ykey $ykey

    if [ $? -eq 0 ]
        then
            echo -e "\n--> Successfully completed cross-validation."
        else
            echo -e "\n--> Error: python script failed during cross-validation. Skipping remaining functions."
            cleanup_tmp_files
            exit 1
    fi

    if [ "$refit" == "True" ]; then
        echo -e "\n --> Ovewriting any passed hyperparams with the new output fom CV"
        hyperparams="${out}/cv/hp_search/${train_prefix}_cvround_${hp_round}_cv_scores.csv"

        if [ ! -s "${hyperparams}" ]; then
            echo "Error: output from hyperparameter search not found in path: ${bst_path}"
            exit 1
        else
            echo -e "Output from hyperparameter search saved in path: ${hyperparams}\n"
        fi
    fi
else
    echo -e "\n######################## SKIPPING CROSS-VALIDATION #############################\n"
fi

################################ RUN XGBOOST REFIT WITH DASK ####################################
# NOTE: Creates the refit model to use for prediction in test data below
# NOTE: There is no point passing the same data to refit and predit as refit automatically runs
# predictions on the train data and saves the output
# NOTE: None of the CV/search args matter because only the refit is run here
# NOTE: The key training HPs (max_depth, n_boost_round, xgb_param_eta, xgb_param_subsample, xgb_param_colsample)
# should all be set carefully, preferably after running the CV script
# NOTE: xgb_model_type can be "orig" (full model) or "shap" (smaller model refit on predictors used in the full model)
# This means that the SHAP model will likely be smaller (fewer predictors) than the full model
# Using "shap" is consistent with all the saved data from the refit on train so is more consistent
# However "orig" is the full model learned from the train data so is more "pure" as it doesn't fit twice
# The "orig" model will also likely run out of memory when computing shap values
#################################################################################################

if [ "$refit" == "True" ]; then
    echo -e "\n############################### RUNNING REFIT ##################################\n"

    if [ ! -s "${out}/refit" ];then
        mkdir ${out}/refit
        echo "Created subdirectory ${out}/refit"
    else
        echo "Sub dir ${out}/refit already exists"
    fi

    python3 ${scriptdir}/refit.py \
        --in_ml $ftrain_tmp \
        --n_threads_per_worker $threads \
        --time_per_worker $time \
        --mem_per_worker $mem \
        --n_workers_in_cluster $workers \
        --n_folds $cv_folds \
        --out $out \
        --prefix ${train_prefix}_refit \
        --local_dir /tmp/${SLURM_JOB_ID} \
        --row_chunk_size $dask_row_chunks \
        --cluster distributed \
        --incremental_learning False \
        --incremental_start_round 0 \
        --incremental_n_boost_per_round 10 \
        --verbose True \
        --xgb_eval_metric logloss \
        --run_shap_main True \
        --run_shap_inter False \
        --hp_search_results $hyperparams \
        --gpu $gpu \
        --interface $interface \
        --xkey $xkey \
        --ykey $ykey \
        --worker_queue $worker_queue

    if [ $? -eq 0 ]
        then
            echo -e "\n--> Successfully completed refit."
        else
            echo -e "\n--> Error: python script failed during refit. Skipping remaining functions."
            cleanup_tmp_files
            exit 1
    fi
else
    echo -e "\n############################### SKIPPING REFIT #################################\n"
fi

############################# RUN XGBOOST PREDICT WITH WITH DASK ###################################
# NOTE: Uses the refit model from above to predict in test data
# NOTE: See notes above about choice of "xgb_model_type" and "time_per_worker"
####################################################################################################

if [ "$predict" == "True" ]; then
    echo -e "\n############################## RUNNING PREDICT #################################\n"

    if [ ! -s "${out}/predict" ]; then
        mkdir ${out}/predict
        echo "Created subdirectory ${out}/predict"
    else
        echo "Sub dir ${out}/predict already exists"
    fi

    if [ -s "${bst}" ]; then
        bst_path="$bst"
        col_path="$bstcols"
    else
        bst_path="${out}/refit/models/${train_prefix}_refit_${xgb_model_type}refit_xgbmodel.json"
        col_path="${out}/refit/predictors/${train_prefix}_refit_used_cols.csv"
    fi

    if [ ! -s "${bst_path}" ]; then
        echo -e "\nError: fit XGB model not found in path: ${bst_path}\n"
        exit 1
    fi

    if [ ! -s "${col_path}" ]; then
        echo -e "\nError: CSV file with columns used in XGB refit not found in path: ${col_path}\n"
        exit 1
    fi

    if [[ "${ykey}" == "y_adjusted" ]]; then
        platt_scale=$(echo ${bst_path/xgbmodel/plattscalemodel})
    else
        platt_scale='none'
    fi

    python3 ${scriptdir}/predict.py \
        --in_ml $ftest_tmp \
        --used_cols ${col_path} \
        --bst_path ${bst_path} \
        --cluster distributed \
        --n_threads_per_worker $threads \
        --time_per_worker $time \
        --mem_per_worker $mem \
        --n_workers_in_cluster $workers \
        --out $out \
        --prefix ${test_prefix}_predict \
        --local_dir /tmp/${SLURM_JOB_ID} \
        --row_chunk_size $dask_row_chunks \
        --run_shap_main True \
        --run_shap_inter $run_shap_inter \
        --platt_scale_model $platt_scale \
        --gpu $gpu \
        --interface $interface \
        --xkey $xkey \
        --ykey $ykey \
        --worker_queue $worker_queue


    if [ $? -eq 0 ]
        then
            echo -e "\n--> Successfully completed prediction."
        else
            echo -e "\n--> Error: python script failed during prediction."
            cleanup_tmp_files
            exit 1
    fi
else
    echo -e "\n############################# SKIPPING PREDICT #################################\n"
fi

cleanup_tmp_files

end=`date +%s`

runtime=$((end-start))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$(( (runtime % 3600) % 60 ))
echo "Total runtime including all CV/refit/predict calls: $hours:$minutes:$seconds (hh:mm:ss)"

echo -e "\n############################## RUN COMPLETE :) #################################\n"
