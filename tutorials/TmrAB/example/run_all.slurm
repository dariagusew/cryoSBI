# various dir
SCRIPTS_=/pasteur/appa/scratch/mbonomi/cryoSBI/src/cryo_sbi/utils
DATA_=/pasteur/appa/scratch/mbonomi/cryoSBI/tutorials/TmrAB/data

# activate conda
source ~/SCRATCH/miniconda3/bin/activate
# activate conda environment
conda activate cryo-EM

# 0. get directory test
jobdir=`sed -n ${SLURM_ARRAY_TASK_ID}p jobdir_list`
# go there and do your job
cd $jobdir

# 1. train image embedding
pretrain_image_embed --image_config simulation_parameters.json --embedding SPATIAL_CRYO --batch_size 256 --embedding_dim 16 --simulation_batch_size 1024 --epochs 100 --lr 2e-4 --real_data_mrc_path ${DATA_}/1p5mln-TmrAB-stack-128-small.mrcs --real_data_fraction 0.1 > log.pretrain

# 2. train flow
train_nle_model_with_finetuning --image_config_file simulation_parameters.json --train_config_file training_parameters_nle.json --epochs 200 --estimator_file tutorial_estimator.pt --n_workers 4 --train_device cuda --pretrained_embedding_path pretrained_image_embed.pt --freeze_embedding True --loss_file tutorial.loss --simulation_batch_size 2048 --validation_mrc_path ${DATA_}/1p5mln-TmrAB-stack-128-small.mrcs --real_data_fraction 0.1 > log.train

# 3. do inference
ln -s TmrAB*.pt TmrAB.pt
run_inference_real_data --image-stack ${DATA_}/1p5mln-TmrAB-stack-128-small.mrcs --models_file TmrAB.pt --estimator_file tutorial_estimator.pt --train_config_file training_parameters_nle.json --image_config_file simulation_parameters.json --output-file results.pt > log.inference
