import os
import tempfile
import torch  # type: ignore
import torchaudio  # type: ignore
import logging
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from mmaudio.eval_utils import (  # type: ignore
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    setup_eval_logging,
)
from mmaudio.model.flow_matching import FlowMatching  # type: ignore
from mmaudio.model.networks import get_my_mmaudio  # type: ignore
from mmaudio.model.utils.features_utils import FeaturesUtils  # type: ignore

# Basic Setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

setup_eval_logging()
log = logging.getLogger()

app = FastAPI()

# Global variables for model
net = None
feature_utils = None
fm = None
seq_cfg = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


@app.on_event("startup")  # type: ignore
def load_model():
    global net, feature_utils, fm, seq_cfg
    log.info("Loading MMAudio model...")
    variant = "large_44k_v2"
    model: ModelConfig = all_model_cfg[variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(
        torch.load(model.model_path, map_location=device, weights_only=True)
    )
    log.info(f"Loaded weights from {model.model_path}")

    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

    feature_utils = (
        FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            synchformer_ckpt=model.synchformer_ckpt,
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        .to(device, dtype)
        .eval()
    )

    log.info("MMAudio model loaded and ready to serve!")


@app.post("/generate_v2a")
async def generate_v2a_endpoint(
    video: UploadFile = File(...),
    prompt: str = Form(""),
    duration: float = Form(8.0),
):
    log.info(f"Received request for prompt: {prompt}, duration: {duration}")

    # Save uploaded video
    fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    with open(tmp_video, "wb") as f:
        content = await video.read()
        f.write(content)

    try:
        video_info = load_video(tmp_video, duration)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        duration = video_info.duration_sec

        assert seq_cfg is not None
        assert net is not None
        seq_cfg.duration = duration
        net.update_seq_lengths(
            seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
        )

        rng = torch.Generator(device=device)
        rng.manual_seed(42)

        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            audios = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[""],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=4.5,
            )
        audio = audios.float().cpu()[0]

        fd_out, tmp_audio = tempfile.mkstemp(suffix=".wav")
        os.close(fd_out)
        torchaudio.save(tmp_audio, audio, seq_cfg.sampling_rate)

        return FileResponse(tmp_audio, media_type="audio/wav")
    except Exception as e:
        log.error(f"Error during generation: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
