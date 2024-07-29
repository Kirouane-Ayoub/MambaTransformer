import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        use_linear_attn: bool = False,
    ):
        """
        Initializes the MultiHeadAttention class.

        Args:
            dim (int): The dimension of the input and output vectors.
            heads (int): The number of attention heads.
            dim_head (int): The dimension of each attention head.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            use_linear_attn (bool, optional): Whether to use linear attention. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.use_linear_attn = use_linear_attn

        inner_dim = heads * dim_head
        self.scale = dim_head**-0.5

        # Query, Key, and Value projection layers
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the MultiQueryAttention model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        b, n, d = x.shape

        q = self.to_q(x).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)

        q *= self.scale

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = attn_output.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, *, heads=4, dim_head=64, dropout=0.0):
        """
        Initializes the LinearAttention class.

        Args:
            dim (int): The dimension of the input and output vectors.
            heads (int, optional): The number of attention heads. Defaults to 4.
            dim_head (int, optional): The dimension of each attention head. Defaults to 64.
            dropout (float, optional): The dropout rate. Defaults to 0.0.

        """
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """
        Forward pass of the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).

        Description:
            This function performs the forward pass of the multi-head attention layer. It takes an input tensor `x` and an optional mask tensor `mask` as input. The input tensor `x` has a shape of (batch_size, sequence_length, dim), where `batch_size` is the number of batches, `sequence_length` is the length of the input sequence, and `dim` is the dimension of the input vectors. The mask tensor `mask` is used to mask out certain positions in the input sequence.

            The function first splits the input tensor `x` into queries, keys, and values using the linear layer `self.to_qkv`. It then reshapes and transposes the queries, keys, and values to prepare them for multi-head attention. The queries, keys, and values are scaled by `self.scale` and then passed through softmax functions to compute attention weights.

            If a mask tensor is provided, it is used to mask out certain positions in the keys using the `masked_fill` function.

            The function then computes the attention context by performing matrix multiplication between the attention weights and the values. The output is computed by performing matrix multiplication between the attention context and the queries.

            Finally, the output tensor is reshaped back to its original dimensions and passed through a linear layer `self.to_out` before being returned.

        """
        h = self.heads
        # Get queries, keys, and values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(q.shape[0], q.shape[1], h, -1).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], h, -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], h, -1).transpose(1, 2)

        q *= self.scale
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)

        if mask is not None:
            k = k.masked_fill(mask, float("-inf"))

        # Compute context and output
        context = torch.einsum("b h n d, b h n e -> b h d e", q, k)
        out = torch.einsum("b h d e, b h n d -> b h n e", context, v)

        # Reshape back to original dimensions and apply the final linear layer
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim: int = None,
        depth: int = 5,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        """
        Initializes the MambaBlock module with the given parameters.

        Args:
            dim (int, optional): The input dimension of the block. Defaults to None.
            depth (int, optional): The number of layers in the block. Defaults to 5.
            d_state (int, optional): The dimension of the state. Defaults to 16.
            expand (int, optional): The multiplier for the inner dimension. Defaults to 2.
            d_conv (int, optional): The kernel size for the 1D convolution. Defaults to 4.
            conv_bias (bool, optional): Whether to include bias in the convolution. Defaults to True.
            bias (bool, optional): Whether to include bias in the linear layers. Defaults to False.

        Returns:
            None

        This function initializes the MambaBlock module with the given parameters. It sets the input dimension,
        number of layers, state dimension, inner dimension multiplier, kernel size for the 1D convolution,
        and whether to include bias in the convolution and linear layers. It also initializes the linear layers,
        convolution, and parameters for the state space model. Finally, it sets the output projection linear layer.

        Note: The dt_rank attribute is set to the ceiling of the dim divided by 16. The dim_inner attribute is set
        to the product of dim and expand. The in_proj linear layer is initialized with dim as input and dim_inner * 2
        as output. The conv1d 1D convolution is initialized with dim_inner as input and output channels, with conv_bias
        as bias, kernel_size of d_conv, groups of dim_inner, and padding of d_conv - 1. The x_proj linear layer is
        initialized with dim_inner as input and dt_rank + self.d_state * 2 as output. The dt_proj linear layer is
        initialized with dt_rank as input and dim_inner as output. The A_log parameter is initialized with the log of
        a tensor of shape (d_state, dim_inner), where d_state is the dimension of the state. The D parameter is
        initialized with a tensor of shape (dim_inner,) filled with ones. The out_proj linear layer is initialized
        with dim_inner as input and dim as output.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = dim * expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(dim, dim_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=dim_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(dim_inner, dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim_inner))
        self.out_proj = nn.Linear(dim_inner, dim, bias=bias)

    def forward(self, x: Tensor):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x_and_res = rearrange(x_and_res, "b l x -> b x l")
        (x, res) = x_and_res.split(split_size=[self.dim_inner, self.dim_inner], dim=1)

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

        return output

    def ssm(self, x: Tensor):
        """
        Runs the State Space Model (SSM) for the given input `x`.

        Args:
            x (Tensor): The input tensor of shape (b, d_in, l), where b is the batch size,
                d_in is the input dimension, and l is the sequence length.

        Returns:
            Tensor: The output tensor of shape (b, d_in, l), representing the state space model output.

        This function computes the state space parameters ∆ A B C D, where A and D are input independent,
        and ∆, B, C are input-dependent. It then uses these parameters to perform a selective scan,
        which is similar to the `run_SSM` function in The Annotated S4 [2]. The output is the state space
        model output for the given input `x`.
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_dbl)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Performs a selective scan using the given parameters to compute the state space model output.

        Args:
            u (Tensor): The input tensor of shape (b, d_in, l), where b is the batch size,
                d_in is the input dimension, and l is the sequence length.
            delta (Tensor): The continuous parameter tensor of shape (b, l, dt_rank).
            A (Tensor): The input-independent state space parameter tensor of shape (d_in, n).
            B (Tensor): The input-dependent state space parameter tensor of shape (b, l, n).
            C (Tensor): The input-dependent state space parameter tensor of shape (b, l, n).
            D (Tensor, optional): The input-independent state space parameter tensor of shape (d_in,).

        Returns:
            Tensor: The output tensor of shape (b, d_in, l), representing the state space model output.
        """
        (b, d_in, l) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (Δ, A, B)  (see Section 2 Equation 4 in the Mamba paper [1])
        # Note that B is parameterized directly
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n")

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = torch.zeros((b, d_in, n), device=next(self.parameters()).device)
        ys = []
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], "b d_in n , b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=2)  # (b d_in l)

        if D is not None:
            y = y + u * rearrange(D, "d_in -> d_in 1")

        return y


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,
        post_act_ln: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        no_bias: Optional[bool] = False,
        triton_kernels_on: bool = False,
    ):
        """
        Initializes a FeedForward module.

        Args:
            dim (Optional[int], optional): The input dimension. Defaults to None.
            dim_out (Optional[int], optional): The output dimension. Defaults to None.
            mult (Optional[int], optional): The multiplier for the inner dimension. Defaults to 4.
            post_act_ln (Optional[bool], optional): Whether to apply layer normalization after the activation function. Defaults to False.
            dropout (Optional[float], optional): The dropout rate. Defaults to 0.0.
            no_bias (Optional[bool], optional): Whether to use bias in the linear layers. Defaults to False.
            triton_kernels_on (bool, optional): Whether to use Triton kernels. Defaults to False.

        Initializes the FeedForward module with the given parameters. The module consists of a linear layer followed by an activation function.
        If `post_act_ln` is True, a layer normalization layer is added after the activation function.
        Finally, a dropout layer and another linear layer are added.
        The output dimension is determined by `dim_out` if provided, otherwise it is set to the input dimension.
        The inner dimension is calculated by multiplying `dim` with `mult`. The activation function used is `nn.SiLU()`.
        """

        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.post_act_ln = post_act_ln
        self.dropout = dropout
        self.no_bias = no_bias
        self.triton_kernels_on = triton_kernels_on

        inner_dim = int(dim * mult)
        dim_out = dim_out or dim  # Default to input dimension if not provided

        # Determine activation function
        activation = nn.SiLU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias), activation
        )

        # Define feedforward network
        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

    def forward(self, x):
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim_out).
        """
        return self.ff(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        """
        Initializes an instance of the class with the given dimension.

        Args:
            dim (int): The dimension of the instance.
        """
        super().__init__()
        self.scale = dim ** (-0.5)
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).

        This function takes an input tensor `x` and applies normalization along the last dimension using the `F.normalize` function.
        It then multiplies the normalized tensor by the product of `self.scale` and `self.g`.
        The output tensor has the same shape as the input tensor.
        """
        return F.normalize(x, dim=-1) * self.scale * self.g


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the TransformerBlock module.

        Args:
            dim (int): The dimensionality of the input and output.
            heads (int): The number of attention heads.
            dim_head (int): The dimensionality of each attention head.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
            use_linear_attn (bool, optional): Whether to use linear attention. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.


        This constructor initializes the TransformerBlock module with the specified parameters. It sets the module's attributes
        such as `dim`, `heads`, `dim_head`, `dropout`, `ff_mult`, and `use_linear_attn`. It also initializes the module's
        submodules, including the `attn`, `linear_attn`, `ffn`, and `norm` submodules. The `attn` submodule is an instance of
        the `MultiQueryAttention` class, and the `linear_attn` submodule is an instance of the `LinearAttention` class. The
        `ffn` submodule is an instance of the `FeedForward` class. The `norm` submodule is an instance of the `nn.LayerNorm`
        class.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.use_linear_attn = use_linear_attn

        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Linear Attention
        self.linear_attn = LinearAttention(
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
        )

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the model.

        This function takes an input tensor `x` and applies the forward pass of the model. If `use_linear_attn` is True, it
        applies linear attention to `x`, normalizes the output using the `norm` module, and then applies the feed-forward
        network (`ffn`) to the normalized output. If `use_linear_attn` is False, it applies the multi-query attention
        (`attn`) to `x`, normalizes the output, and then applies the feed-forward network (`ffn`). The output tensor is
        returned.
        """

        if self.use_linear_attn:
            x = self.linear_attn(x)
            x = self.norm(x)
            x = self.ffn(x)
        else:
            x, _, _ = self.attn(x)
            x = self.norm(x)
            x = self.ffn(x)

        return x


class MambaTransformerblock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the MambaTransformerblock.

        Args:
            dim (int): The dimensionality of the token embeddings and model hidden states.
            heads (int): The number of attention heads.
            depth (int): The number of transformer blocks.
            dim_head (int): The dimensionality of each attention head.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
            d_state (int, optional): The dimensionality of the state embeddings. Defaults to None.
            transformer_depth (int, optional): The number of transformer blocks. Defaults to 1.
            mamba_depth (int, optional): The number of Mamba blocks. Defaults to 1.
            use_linear_attn (bool, optional): Whether to use linear attention. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        # Mamba, Transformer, and ffn blocks
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(dim, mamba_depth, d_state, *args, **kwargs)
                for _ in range(mamba_depth)
            ]
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    dropout,
                    ff_mult,
                    use_linear_attn,
                    *args,
                    **kwargs,
                )
                for _ in range(transformer_depth)
            ]
        )

        self.ffn_blocks = nn.ModuleList(
            [FeedForward(dim, dim, ff_mult, *args, **kwargs) for _ in range(depth)]
        )

        # Layernorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaTransformerblock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the MambaTransformerblock.

        This function applies the forward pass of the MambaTransformerblock to the input tensor `x`. It iterates over the
        `mamba_blocks`, `transformer_blocks`, and `ffn_blocks` in parallel using the `zip` function. For each iteration,
        it applies the `mamba` block to the input tensor `x`, adds the result back to `x`, applies the `attn` block to the
        updated `x`, adds the result back to `x`, applies the `ffn` block to the updated `x`, and adds the result back to
        `x`. The process is repeated for each block in the respective lists. Finally, the function returns the output
        tensor `x` after passing through all the blocks.

        Note: The `norm` function is called before and after each block to normalize the tensor `x`.

        Example usage:
            input_tensor = torch.randn(16, 32, 512)  # Input tensor of shape (batch_size, sequence_length, hidden_size)
            output_tensor = model(input_tensor)  # Forward pass through the MambaTransformerblock
        """
        for mamba, attn, ffn in zip(
            self.mamba_blocks,
            self.transformer_blocks,
            self.ffn_blocks,
        ):
            x = self.norm(x)
            x = mamba(x) + x
            x = self.norm(x)
            x = attn(x) + x
            x = self.norm(x)
            x = ffn(x) + x

        return x


class MambaTransformer(nn.Module):
    """
    MambaTransformer is a PyTorch module that implements the Mamba Transformer model.

    Args:
        num_tokens (int): The number of tokens in the input vocabulary.
        dim (int): The dimensionality of the token embeddings and model hidden states.
        heads (int): The number of attention heads.
        depth (int): The number of transformer blocks.
        dim_head (int): The dimensionality of each attention head.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.
        d_state (int, optional): The dimensionality of the state embeddings. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        d_state: int = None,
        return_embeddings: bool = False,
        transformer_depth: int = 1,
        mamba_depth: int = 1,
        use_linear_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.d_state = d_state
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.d_state = d_state
        self.return_embeddings = return_embeddings
        self.transformer_depth = transformer_depth
        self.mamba_depth = mamba_depth

        self.emb = nn.Embedding(num_tokens, dim)
        self.mt_block = MambaTransformerblock(
            dim,
            heads,
            depth,
            dim_head,
            dropout,
            ff_mult,
            d_state,
            return_embeddings,
            transformer_depth,
            mamba_depth,
            use_linear_attn,
            *args,
            **kwargs,
        )
        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaTransformer model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, num_tokens).
        """
        x = self.emb(x)
        x = self.mt_block(x)

        if self.return_embeddings:
            return x

        else:
            return self.to_logits(x)
