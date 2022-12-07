

encoder_ckpt_path = 'work_dirs/pre-trained/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth'  # noqa

stylegan_weights = 'work_dirs/experiments/res/9_Snowman/ckpt/9_Snowman/iter_400.pth'  # noqa

model = dict(type='PSPEncoderDecoder',
             encoder=dict(type='VAEStyleEncoder',
                          num_layers=50,
                          pretrained=dict(ckpt_path=encoder_ckpt_path,
                                          prefix='encoder',
                                          strict=False)),
             decoder=dict(type='SwapStyleGANv2Generator',
                          out_size=1024,
                          style_channels=512,
                          num_mlps=8,
                          pretrained=dict(ckpt_path=stylegan_weights,
                                          prefix='generator_ema')),
             pool_size=(1024, 1024),
             id_lambda=0.0,
             lpips_lambda=0.0,
             id_ckpt=None,
             kl_loss=None,
             train_cfg=None,
             test_cfg=None)

train_cfg = None
test_cfg = None
