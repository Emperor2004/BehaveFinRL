import importlib
mods = ['config', 'utils.logger', 'utils.save_utils', 'utils.framework', 'train']
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m} imported successfully")
    except Exception as e:
        print(f"Error importing {m}: {e}")
