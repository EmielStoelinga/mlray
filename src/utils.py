from ray.data.preprocessors import Categorizer
import pandas as pd


def get_wachttijden_categorizer():
    return Categorizer(
        columns=[
            'TYPE_WACHTTIJD',
            'SPECIALISME',
            'ROAZ_REGIO',
            'TYPE_ZORGINSTELLING',
        ],
        dtypes={
            'TYPE_WACHTTIJD': pd.CategoricalDtype(['Polikliniekbezoek', 'Diagnostiek', 'Behandeling']),
            'SPECIALISME': pd.CategoricalDtype(['Dermatologie (310)', 'Neurologie (330)',
                                                'Maag, darm en leverziekten (318)', 'Orthopedie (305)',
                                                'Anesthesiologie (389)', 'Radiologie (362)', 'Reumatologie (324)',
                                                'Chirurgie (heelkunde) (303)', 'Oogheelkunde (301)',
                                                'Orthopedie (305) / Chirurgie (heelkunde) (303)',
                                                'Neurochirurgie (308)', 'KNO (302)', 'Plastische Chirurgie (304)',
                                                'Interne geneeskunde (313)',
                                                'Orthopedie (305) / Neurochirurgie (308)',
                                                'Urologie (306) / Chirurgie (heelkunde) (303)',
                                                'Revalidatie (327)', 'Geriatrie (335)', 'Gynaecologie (307)',
                                                'Urologie (306)',
                                                'Chirurgie (heelkunde) (303) / Plastische Chirurgie (304) / Orthopedie (305)',
                                                'Cardiologie (320)', 'Revalidatiegeneeskunde (327)',
                                                'Kindergeneeskunde (316)', 'Sportgeneeskunde', 'Kaakchirurgie',
                                                'Longgeneeskunde (322)',
                                                'Dermatologie (310) / Chirurgie (heelkunde) (303)',
                                                'Psychiatrie (329)',
                                                'KNO (302) / Longgeneeskunde (322) / Neurologie (330)', 'Overige',
                                                'Cardiopulmonale chirurgie (328)']),
            'ROAZ_REGIO': pd.CategoricalDtype(['Oost', 'Zuidwest-NL', 'Limburg', 'Zwolle', 'Midden-NL',
                                               'SpzNet AMC', 'West', 'Noordwest', 'Noord-NL', 'Euregio',
                                               'Brabant']),
            'TYPE_ZORGINSTELLING': pd.CategoricalDtype(['Ziekenhuis', 'Kliniek']),
        })
