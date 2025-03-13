type ImageGenerationModelType = {
  architecture: string;
  textToImage: boolean;
  imageToImage: boolean;
  loraSupport: boolean;
  links: string[];
};

export const IMAGE_GENERATION_MODELS: ImageGenerationModelType[] = [
  {
    architecture: 'Latent Consistency Model',
    textToImage: true,
    imageToImage: true,
    loraSupport: true,
    links: ['https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7'],
  },
  {
    architecture: 'Stable Diffusion',
    textToImage: true,
    imageToImage: true,
    loraSupport: true,
    links: [
      'https://huggingface.co/CompVis/stable-diffusion-v1-1',
      'https://huggingface.co/CompVis/stable-diffusion-v1-2',
      'https://huggingface.co/CompVis/stable-diffusion-v1-3',
      'https://huggingface.co/CompVis/stable-diffusion-v1-4',
      'https://huggingface.co/junnyu/stable-diffusion-v1-4-paddle',
      'https://huggingface.co/jcplus/stable-diffusion-v1-5',
      'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5',
      'https://huggingface.co/botp/stable-diffusion-v1-5',
      'https://huggingface.co/dreamlike-art/dreamlike-anime-1.0',
      'https://huggingface.co/stabilityai/stable-diffusion-2',
      'https://huggingface.co/stabilityai/stable-diffusion-2-base',
      'https://huggingface.co/stabilityai/stable-diffusion-2-1',
      'https://huggingface.co/bguisard/stable-diffusion-nano-2-1',
      'https://huggingface.co/justinpinkney/pokemon-stable-diffusion',
      'https://huggingface.co/stablediffusionapi/architecture-tuned-model',
      'https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1',
      'https://huggingface.co/ZeroCool94/stable-diffusion-v1-5',
      'https://huggingface.co/pcuenq/stable-diffusion-v1-4',
      'https://huggingface.co/rinna/japanese-stable-diffusion',
      'https://huggingface.co/benjamin-paine/stable-diffusion-v1-5',
      'https://huggingface.co/philschmid/stable-diffusion-v1-4-endpoints',
      'https://huggingface.co/naclbit/trinart_stable_diffusion_v2',
      'https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model',
    ],
  },
  {
    architecture: 'Stable Diffusion XL',
    textToImage: true,
    imageToImage: true,
    loraSupport: true,
    links: [
      'https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9',
      'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0',
      'https://huggingface.co/stabilityai/sdxl-turbo',
    ],
  },
  {
    architecture: 'Stable Diffusion 3',
    textToImage: true,
    imageToImage: false,
    loraSupport: false,
    links: [
      'https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers',
      'https://huggingface.co/stabilityai/stable-diffusion-3.5-medium',
      'https://huggingface.co/stabilityai/stable-diffusion-3.5-large',
      'https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo',
    ],
  },
  {
    architecture: 'Flux',
    textToImage: true,
    imageToImage: false,
    loraSupport: false,
    links: [
      'https://huggingface.co/black-forest-labs/FLUX.1-schnell',
      'https://huggingface.co/Freepik/flux.1-lite-8B-alpha',
      'https://huggingface.co/black-forest-labs/FLUX.1-dev',
      'https://huggingface.co/shuttleai/shuttle-3-diffusion',
      'https://huggingface.co/shuttleai/shuttle-3.1-aesthetic',
      'https://huggingface.co/Shakker-Labs/AWPortrait-FL',
    ],
  },
];
