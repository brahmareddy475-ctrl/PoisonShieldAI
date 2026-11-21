from pathlib import Path
import pandas as pd
from poisonshield.data.loaders import load_raw, synthetic_flows
from poisonshield.data.preprocessing import prepare_dataset


def test_toniot_network_loader_with_binary_and_multiclass_labels(tmp_path):
    data_dir = tmp_path/'toniot'
    data_dir.mkdir()
    sample = synthetic_flows(500, 1).drop(columns=['Source IP','Timestamp'])
    sample['type'] = sample.pop('Label')
    sample['label'] = (sample.type != 'normal').astype(int)
    sample['dns_AA'] = ['T', 'F', '-','T','F']*100
    sample['ssl_resumed'] = ['T','F']*250
    sample.to_csv(data_dir/'Train_Test_Network.csv', index=False)
    sample[['type']].to_csv(data_dir/'telemetry.csv', index=False)  # not network data
    loaded, src = load_raw('toniot', str(data_dir), max_rows_per_file=450)
    assert len(loaded) == 450 and len(src) == 1
    assert set(loaded.dns_AA.unique()) == {0., 1.}
    prepared = prepare_dataset(loaded, 'toniot', 'multiclass')
    assert prepared.classes == ['normal','dos','recon']
    assert 'label' not in prepared.schema.columns
    assert 'type' not in prepared.schema.columns


def test_unswnb15_official_pairs(tmp_path):
    root = tmp_path/'unswnb15'
    root.mkdir()
    data = synthetic_flows(500, 0).drop(columns=['Source IP','Timestamp'])
    data['attack_cat'] = data.pop('Label')
    data['label'] = (data.attack_cat != 'normal').astype(int)
    data.iloc[:300].to_csv(root/'UNSW_NB15_training-set.csv', index=False)
    data.iloc[300:].to_csv(root/'UNSW_NB15_testing-set.csv', index=False)
    raw, paths = load_raw('unswnb15', str(root))
    assert len(raw) == 500 and len(paths) == 2
    prepared = prepare_dataset(raw, 'unswnb15', 'binary')
    assert len(prepared.test.y) == 50
    assert prepared.classes == ['normal','attack']


def test_cic_protocol_is_one_hot(tmp_path):
    root=tmp_path/'cicids2017'
    root.mkdir()
    data=synthetic_flows(500).drop(columns=['proto'])
    data['Protocol'] = [6, 17, 1, 6, 17]*100
    data.to_csv(root/'Monday-WorkingHours.pcap_ISCX.csv', index=False)
    raw, paths=load_raw('cicids2017',str(root))
    prep=prepare_dataset(raw,'cicids2017','binary')
    assert 'Protocol' in prep.schema.categorical_cols
    assert len(next(v for k,v in prep.schema.categories.items() if k == 'Protocol')) == 3 # 3 observed protocols; no unnecessary __other__
