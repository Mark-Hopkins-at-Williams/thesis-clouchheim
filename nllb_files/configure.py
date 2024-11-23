USE_CUDA=True

LOG_FILE = 'results/model_logs.csv'
TRAINING_NOTES = 'encrypted english (gne_Test) gen test, with <gen> encrypted monolingual addition'

# hñähñu (worst performing lang)

AMERICAS_NLP_CODE_TO_LANG = {
    "cni": "ashaninka", 
    "bzd": "bribri", 
    "grn": "guarani",  
    "quy": "quechua",  
    "aym": "aymara", 
    "shp": "shipibo_konibo", 
    "ctp": "chatino", 
    "oto": "hñähñu", 
    "nah": "nahuatl",
    "tar": "raramuri",
    "hch": "wixarika",
    "eng": "english",
    "spa": "spanish",
    "gne": "encypted english"
}

NLLB_SEED_CSV = '/mnt/storage/hopkins/data/nllb/seed/nllb_seed.csv'
NLLB_SEED_LANGS = ['pbt_Arab', 'bho_Deva', 'nus_Latn', 'ban_Latn', 'dzo_Tibt', 'mni_Beng', 'lim_Latn',
                   'ltg_Latn', 'ace_Latn', 'crh_Latn', 'srd_Latn', 'taq_Latn', 'mri_Latn', 'ary_Arab',
                   'bam_Latn', 'knc_Arab', 'eng_Latn', 'knc_Latn', 'dik_Latn', 'prs_Arab', 'bjn_Arab',
                   'vec_Latn', 'fur_Latn', 'kas_Deva', 'kas_Arab', 'arz_Arab', 'lij_Latn', 'ace_Arab',
                   'bjn_Latn', 'scn_Latn', 'bug_Latn', 'lmo_Latn', 'szl_Latn', 'hne_Deva', 'fuv_Latn',
                   'taq_Tfng', 'shn_Mymr', 'mag_Deva']
NLLB_SEED_LPS = [(src, 'eng_Latn') for src in NLLB_SEED_LANGS if src != 'eng_Latn']

AMERICAS_NLP_CSV = '/mnt/storage/clouchheim/thesis-clouchheim/nllb_files/data/americas_nlp_data.csv'

# For multi lingual training all 11

ALL_LANGS = ["cni_Latn", "bzd_Latn", "grn_Latn", "quy_Latn", "aym_Latn", "shp_Latn",
                      "ctp_Latn", "oto_Latn", "nah_Latn", "tar_Latn", "hch_Latn"]


#AMERICAS_NLP_LANGS = ["cni_Latn", "bzd_Latn", "grn_Latn", "quy_Latn", "aym_Latn", "shp_Latn",
                      #"ctp_Latn", "oto_Latn", "nah_Latn", "tar_Latn", "hch_Latn"]

# For bilingual or more language training
AMERICAS_NLP_LANGS = ['gne_Test'] # REMEMBER TO SET DEV LANAGUAGE IN RUN.SH
AMERICAS_NLP_LPS = [('spa_Latn', tgt) for tgt in AMERICAS_NLP_LANGS]

