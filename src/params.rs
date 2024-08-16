use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        //let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        //let params = LLamaParams::from_safetensors(&safetensor, &config);
        /*
            pub(crate) struct LlamaConfigJson {
                pub bos_token_id: u32,   // 起始符token id
                pub eos_token_id: u32,   // 结束符token id 
                pub hidden_size: usize,             // 隐藏层大小，即各层输出的最后一维
                pub intermediate_size: usize,       // Feed-Forward网络的中间层大小
                pub max_position_embeddings: usize, // 最大序列长度
                pub num_attention_heads: usize,     // Self-Attention的Q头数
                pub num_hidden_layers: usize,       // 隐藏层数
                pub num_key_value_heads: usize,     // Self-Attention的{K/V}头数
                pub vocab_size: usize,              // 词表大小
                #[serde(default = "default_rms_norm_eps")]
                pub rms_norm_eps: f32,  // RMS Normalization的epsilon参数
                #[serde(default = "default_rope_theta")]
                pub rope_theta: f32,    // RoPE的theta参数
                pub torch_dtype: String,    // 模型数据类型
                #[serde(default = "default_tie_word_embeddings")]
                pub tie_word_embeddings: bool,  //起始和结束embedding参数矩阵是否共享同一份数据
            }
         */
        //println!("{:?}", safetensor.names());
        let get_tensor = |name: &str| {
            let view = safetensor.tensor(name).unwrap();
            let data: Vec<f32> = view.data()
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(bytes)
                })
                .collect();
            Tensor::new(data, &view.shape().to_vec())
        };

        let layers = config.num_hidden_layers;
        Self {
            // token_id to embedding lookup table
            embedding_table: get_tensor("lm_head.weight"),
            // decoder layer
            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),
            // ffn layer
            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),            
            // output
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
