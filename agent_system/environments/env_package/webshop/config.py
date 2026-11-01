"""Resolve matching WebShop product files and Lucene index selection."""
import os


def build_webshop_kwargs(use_small: bool, human_goals: bool, data_dir: str) -> dict:
    suffix = '_1000' if use_small else ''
    return {
        'observation_mode': 'text',
        'num_products': 1000 if use_small else None,
        'human_goals': human_goals,
        'file_path': os.path.join(data_dir, f'items_shuffle{suffix}.json'),
        'attr_path': os.path.join(data_dir, f'items_ins_v2{suffix}.json'),
    }
