# various dir
SCRIPTS_=/pasteur/helix/scratch/mbonomi/cryoSBI/src/cryo_sbi/utils
DATA_=/pasteur/helix/scratch/mbonomi/cryoSBI/tutorials/RhlR-PqsE/data/mixed-stacks

# activate conda
source ~/SCRATCH/miniconda3/bin/activate
# activate conda environment
conda activate cryo-EM

# 1. train image embedding
python ${SCRIPTS_}/pretrain_image_embed_v7.py --image_config simulation_parameters.json --embedding SPATIAL_CRYO --batch_size 512 --embedding_dim 16 --simulation_batch_size 1024 --epochs 100 --lr 2e-4 --beta 1.0e-5 --beta_NRE 0.1 --beta_cons 0.1 --randomize_SNR --use_Cosine_consistency_loss --real_data_mrc ${DATA_}/small_128_cropfrom_200_flipped.mrc --val_size 3000 > log.pretrain

# 2. do inference
for file in pretrained_image_embed_epoch???.pt 
do
 ii=`echo $file | cut -d_ -f4 | cut -d. -f1 | sed -e "s/epoch//g"`
 python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-0.0.mrc --models_file RhlR-PqsE-allatom.pt --full_model $file --image_config simulation_parameters.json --output_file results-0.0-${ii}.pt --normalize_images > log.inference-small-0.0.$ii
 python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-0.5.mrc --models_file RhlR-PqsE-allatom.pt --full_model $file --image_config simulation_parameters.json --output_file results-0.5-${ii}.pt --normalize_images > log.inference-small-0.5.$ii
 python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-1.0.mrc --models_file RhlR-PqsE-allatom.pt --full_model $file --image_config simulation_parameters.json --output_file results-1.0-${ii}.pt --normalize_images > log.inference-small-1.0.$ii
done
python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-0.0.mrc --models_file RhlR-PqsE-allatom.pt --full_model pretrained_image_embed.pt --image_config simulation_parameters.json --output_file results-0.0.pt --normalize_images > log.inference-small-0.0
python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-0.5.mrc --models_file RhlR-PqsE-allatom.pt --full_model pretrained_image_embed.pt --image_config simulation_parameters.json --output_file results-0.5.pt --normalize_images > log.inference-small-0.5
python ${SCRIPTS_}/infer_populations_nre.py --embedding SPATIAL_CRYO --embedding_dim 16 --image_stack ${DATA_}/mixed-1.0.mrc --models_file RhlR-PqsE-allatom.pt --full_model pretrained_image_embed.pt --image_config simulation_parameters.json --output_file results-1.0.pt --normalize_images > log.inference-small-1.0
