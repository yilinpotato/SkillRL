"""Small product files must select the small Lucene index in the real factory."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_factory_product_index_alignment():
    spec = importlib.util.spec_from_file_location('webshop_config', ROOT / 'agent_system/environments/env_package/webshop/config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execute the production index selector with a recording searcher, no JVM.
    source = ROOT / 'agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'init_search_engine')
    import os
    scope = {'os': os, 'BASE_DIR': '/webshop/web_agent_site', 'LuceneSearcher': lambda p: p}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    for small, suffix, index in [(True, '_1000', 'indexes_1k'), (False, '', 'indexes')]:
        kwargs = module.build_webshop_kwargs(small, False, '/external/data')
        assert Path(kwargs['file_path']).name == f'items_shuffle{suffix}.json'
        assert Path(kwargs['attr_path']).name == f'items_ins_v2{suffix}.json'
        assert Path(scope['init_search_engine'](kwargs['num_products'])).name == index
        assert kwargs['human_goals'] is False
        assert kwargs['observation_mode'] == 'text'


if __name__ == '__main__':
    test_factory_product_index_alignment()
    print('WEB_SHOP_PRODUCT_INDEX_ALIGNMENT_OK')
