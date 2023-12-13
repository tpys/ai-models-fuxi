# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from collections import defaultdict
from ai_models.model import Model

import numpy as np
import xarray as xr
import pandas as pd

LOG = logging.getLogger(__name__)


def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t*freq) for t in [i-1, i, i+1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year/366, p.hour/24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)



class FuXi(Model):
    expver = "fuxi"
    use_an = False
    debug_fx = False

    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fuxi/{file}"
    )
    download_files = [
        "short",
        "short.onnx",
        "meidum",
        "meidum.onnx",     
        "long",
        "long.onnx",             
    ]
    
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["2t", "10u", "10v", "msl", "tp"]
    param_level_pl = (
        ["z", "t", "u", "v", "r"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.hour_steps = 6
        self.lagged = [-6, 0]
        self.stages = ["short", "medium", "long"]

        self.ordering = [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ] + self.param_sfc


    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)
        r["stream"] = {0: "oper", 6: "scda", 12: "oper", 18: "scda"}[time]


    def load_model(self):
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads
        models = {}
        for stage in self.stages:
            model_file = os.path.join(self.assets, f"{stage}.onnx")
            os.stat(model_file)
            with self.timer(f"Loading {model_file}"):
                model = ort.InferenceSession(
                    model_file,
                    sess_options=options,
                    providers=self.providers,
                )
                models[stage] = model
        return models
    
    def get_init_time(self):
        init_time = self.all_fields.order_by(valid_datetime="descending")[0].datetime()
        init_time = pd.to_datetime(init_time)
        return init_time

    def create_input(self, init_time):
        hist_time = init_time - pd.Timedelta(hours=6)
        valid_time = [hist_time, init_time]
        valid_time_str = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in valid_time]

        param_sfc = self.param_sfc
        param_pl, level = self.param_level_pl
        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")        

        fields_pl = fields_pl.sel(valid_datetime=valid_time_str, param=param_pl, level=level)
        fields_pl = fields_pl.order_by(param=param_pl, valid_datetime=valid_time_str, level=level)

        pl = defaultdict(list)
        for field in fields_pl:
            pl[field.metadata("param")].append(field)

        fields_sfc = fields_sfc.sel(valid_datetime=valid_time_str, param=param_sfc)
        fields_sfc = fields_sfc.order_by(param=param_sfc, valid_datetime=valid_time_str)

        sfc = defaultdict(list)
        for field in fields_sfc:
            sfc[field.metadata("param")].append(field)

        input = []
        for param, fields in pl.items():
            data = np.stack(
                [field.to_numpy(dtype=np.float32) for field in fields]
            ).reshape(-1, len(level), len(lat), len(lon))
            input.append(data)
            info = (f"Name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}")
            LOG.info(info)

        for param, fields in sfc.items():
            data = np.stack(
                [field.to_numpy(dtype=np.float32) for field in fields]
            ).reshape(-1, 1, len(lat), len(lon))
            input.append(data)
            info = (f"Name: {param}, shape: {data.shape}, range: {data.min():.3f} ~ {data.max():.3f}")
            LOG.info(info)
        
        input = np.concatenate(input, axis=1)

        if self.debug_fx:
            self.input_xr = xr.DataArray(
                data=input,
                coords=dict(
                    time=valid_time,
                    channel=self.ordering,
                    lat=lat,
                    lon=lon,
                ),
            )
            save_name = valid_time[-1].strftime("%Y%m%d%H.nc")
            self.input_xr.to_netcdf(save_name)
        
        self.template_pl = fields_pl.sel(valid_datetime=valid_time_str[-1])
        self.template_sfc = fields_sfc.sel(valid_datetime=valid_time_str[-1])
        return input[None]


    def run(self):
        total_step = self.lead_time // self.hour_steps

        models = self.load_model()
        init_time = self.get_init_time()
        tembs = time_encoding(init_time, total_step)  
        input = self.create_input(init_time)
        
        with self.stepper(6) as stepper:
            for i in range(total_step):
                step = (i + 1) * self.hour_steps
                stage = self.stages[min(2, i//20)] 

                new_input, = models[stage].run(
                    None, {'input': input, 'temb': tembs[i]}
                )

                pl_chans = len(self.ordering) - len(self.param_sfc)
                pl_data = new_input[0, -1, :pl_chans]
                sfc_data = new_input[0, -1, pl_chans:]

                for data, f in zip(pl_data, self.template_pl):
                    self.write(data, template=f, step=step)

                for data, f in zip(sfc_data, self.template_sfc):
                    self.write(data, template=f, step=step)

                if self.debug_fx:
                    ds = xr.DataArray(
                        data=new_input[:, -1], 
                        coords=dict(
                            step=[step],
                            channel=self.input_xr.channel,
                            lat=self.input_xr.lat,
                            lon=self.input_xr.lon,
                        ),
                    )
                    save_dir = init_time.strftime("%Y%m%d%H")
                    os.makedirs(save_dir, exist_ok=True)
                    ds.to_netcdf(os.path.join(save_dir, f"{step:03d}.nc"))  

                stepper(i, step)
                input = new_input
