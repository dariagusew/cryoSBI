import torch


# ============================================================================
# LOW-LEVEL CTF GENERATION
# ============================================================================

def generate_ctf(
    num_pixels: int,
    pixel_size: float,
    defocus_u: torch.Tensor,   # (N,) in Å, positive = underfocus
    defocus_v: torch.Tensor,   # (N,) in Å
    defocus_angle: torch.Tensor,  # (N,) in degrees (astigmatism angle)
    voltage: torch.Tensor,     # (N,) in kV
    cs: torch.Tensor,          # (N,) in mm
    amplitude_contrast: torch.Tensor,  # (N,) fraction, e.g. 0.1
    b_factor: torch.Tensor,    # (N,) in Å²
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Generate a batch of 2-D CTF images.

    Uses the same sign convention as the existing :func:`apply_ctf`:
        CTF = -√(1 - A²) · sin(γ) - A · cos(γ)

    where the phase aberration function is:
        γ(k, φ) = π · λ · Δf(φ) · k²  -  ½π · Cs · λ³ · k⁴

    and the astigmatic defocus is:
        Δf(φ) = (ΔfU + ΔfV) / 2  +  (ΔfU - ΔfV) / 2 · cos(2(φ - θ_ast))

    The frequency grid is built with :func:`torch.fft.fftfreq` (DC at corner),
    matching the layout expected by :func:`torch.fft.fft2` / :func:`torch.fft.ifft2`.

    Args:
        num_pixels:         Side length of the square image (pixels).
        pixel_size:         Pixel size in Å.
        defocus_u:          Defocus along U axis, shape (N,), in Å.
        defocus_v:          Defocus along V axis, shape (N,), in Å.
        defocus_angle:      Astigmatism angle, shape (N,), in degrees.
        voltage:            Accelerating voltage, shape (N,), in kV.
        cs:                 Spherical aberration, shape (N,), in mm.
        amplitude_contrast: Amplitude contrast ratio, shape (N,).
        b_factor:           B-factor envelope, shape (N,), in Å².
        device:             Torch device string.

    Returns:
        CTF tensor of shape (N, num_pixels, num_pixels), dtype float32.
        The DC component is at the corner (standard fft2 layout).
    """
    N = defocus_u.shape[0]

    # ---- frequency grid (cycles / Å), DC at corner (fft2 layout) ----------
    freq_1d = torch.fft.fftfreq(num_pixels, d=pixel_size, device=device)  # (P,)
    # ky along rows (dim 0), kx along columns (dim 1)
    ky, kx = torch.meshgrid(freq_1d, freq_1d, indexing='ij')  # (P, P)
    k2 = kx ** 2 + ky ** 2                                     # (P, P)  [1/Å²]
    phi = torch.atan2(ky, kx)                                  # (P, P)  azimuthal angle

    # ---- relativistic wavelength (Å) ---------------------------------------
    # λ = 12.2643247 / sqrt(V_kV * 1000 + 0.978466 * V_kV²)
    # (same formula as the existing apply_ctf)
    lam = 12.2643247 / torch.sqrt(
        voltage * 1e3 + 0.978466 * voltage ** 2
    )  # (N,)

    # ---- unit conversions --------------------------------------------------
    cs_angstrom = cs * 1e7                                  # mm → Å  (N,)
    angle_rad   = defocus_angle * (torch.pi / 180.0)        # deg → rad  (N,)

    # ---- broadcast: (N,) → (N, 1, 1) for grid operations ------------------
    def _b(t):
        return t.view(-1, 1, 1)

    lam_b   = _b(lam)
    cs_b    = _b(cs_angstrom)
    amp_b   = _b(amplitude_contrast)
    bf_b    = _b(b_factor)
    dfu_b   = _b(defocus_u)
    dfv_b   = _b(defocus_v)
    ang_b   = _b(angle_rad)

    k2_b  = k2.unsqueeze(0)   # (1, P, P)
    phi_b = phi.unsqueeze(0)  # (1, P, P)

    # ---- astigmatic defocus Δf(φ) ------------------------------------------
    delta_f = (
        (dfu_b + dfv_b) / 2.0
        + (dfu_b - dfv_b) / 2.0 * torch.cos(2.0 * (phi_b - ang_b))
    )  # (N, P, P)  in Å

    # ---- phase aberration γ -------------------------------------------------
    gamma = (
        torch.pi * lam_b * delta_f * k2_b
        - 0.5 * torch.pi * cs_b * lam_b ** 3 * k2_b ** 2
    )  # (N, P, P)

    # ---- B-factor envelope --------------------------------------------------
    envelope = torch.exp(-bf_b * k2_b * 0.25)  # (N, P, P)

    # ---- CTF (same sign convention as existing apply_ctf) ------------------
    ctf = envelope * (
        -torch.sqrt(torch.clamp(1.0 - amp_b ** 2, min=0.0)) * torch.sin(gamma)
        - amp_b * torch.cos(gamma)
    )  # (N, P, P)

    return ctf  # float32, DC at corner


def apply_ctf_batch(
    images: torch.Tensor,
    ctf: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a CTF to a batch of images in Fourier space.

    Args:
        images: Real-space images, shape (N, H, W).
        ctf:    CTF tensor from :func:`generate_ctf`, shape (N, H, W).
                DC must be at the corner (standard fft2 layout).

    Returns:
        CTF-modulated images, shape (N, H, W), real-space.
    """
    F = torch.fft.fft2(images)          # (N, H, W) complex
    F_ctf = F * ctf                     # element-wise multiply
    return torch.fft.ifft2(F_ctf).real  # (N, H, W)


def remove_ctf_batch(
    images: torch.Tensor,
    ctf: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Remove (deconvolve) the CTF from a batch of images using a Wiener filter.

    The correction applied is:
        F_corr(k) = F_image(k) · CTF(k) / (CTF(k)² + ε)

    The regularisation parameter ε prevents division by zero at CTF zeros and
    controls the noise amplification.  A value of 1e-3 is a reasonable default
    for typical cryo-EM data; increase it if the output is too noisy.

    Args:
        images: Real-space images, shape (N, H, W).
        ctf:    CTF tensor from :func:`generate_ctf`, shape (N, H, W).
                DC must be at the corner (standard fft2 layout).
        eps:    Wiener regularisation parameter (default: 1e-3).

    Returns:
        CTF-corrected images, shape (N, H, W), real-space.
    """
    F = torch.fft.fft2(images)                  # (N, H, W) complex
    ctf2 = ctf ** 2                             # (N, H, W)
    F_corr = F * ctf / (ctf2 + eps)            # Wiener filter
    return torch.fft.ifft2(F_corr).real         # (N, H, W)


# ============================================================================
# LEGACY FUNCTION (kept for backward compatibility)
# ============================================================================

def apply_ctf(
    image: torch.Tensor,
    defocus: torch.Tensor,  # Defocus in µm (positive = underfocus)
    b_factor: torch.Tensor,  # in Å²
    amp: torch.Tensor,  # amplitude contrast (0.07-0.1 typical)
    simulation_param: dict # simulation parameters
) -> torch.Tensor:
    """
    Applies the CTF to the image in Fourier space.

    Legacy wrapper around :func:`generate_ctf` and :func:`apply_ctf_batch`.

    Args:
        image: Input image (num_batch, num_pixels, num_pixels)
        defocus: Defocus in micrometers. Can be a tensor of shape [batch_size, 1]
                 (for no astigmatism) or [batch_size, 3] to include astigmatism
                 in the format [defocus, defocus_astig, defocus_astig_angle].
        b_factor: B-factor in Å²
        amp: Amplitude contrast ratio (typically 0.07-0.1)
        simulation_param: Simulation parameters

    Returns:
        Image with CTF applied
    """
    device = simulation_param["device"]

    if isinstance(simulation_param["num_pixels"], torch.Tensor):
        num_pixels = int(simulation_param["num_pixels"].item())
    else:
        num_pixels = int(simulation_param["num_pixels"])

    if isinstance(simulation_param["pixel_size"], torch.Tensor):
        pixel_size = simulation_param["pixel_size"].item()
    else:
        pixel_size = simulation_param["pixel_size"]

    N = image.shape[0]
    voltage = torch.full((N,), simulation_param["voltage"], dtype=torch.float32, device=device)
    cs      = torch.full((N,), simulation_param["cs"],      dtype=torch.float32, device=device)

    # Build defocus_u, defocus_v, defocus_angle (all in Å / degrees)
    if simulation_param["astigmatism"]:
        # defocus shape: [N, 3] -> [defocus_avg_um, defocus_astig_um, angle_deg]
        defocus_avg_A   = defocus[:, 0] * 1e4      # µm → Å
        defocus_astig_A = defocus[:, 1] * 1e4      # µm → Å
        defocus_angle   = defocus[:, 2]             # degrees
        defocus_u = defocus_avg_A + defocus_astig_A
        defocus_v = defocus_avg_A - defocus_astig_A
    elif defocus.ndim == 2 and defocus.shape[1] == 3:
        # Non-astigmatic case with star-file prior: use first column only
        defocus_u     = defocus[:, 0] * 1e4        # µm → Å
        defocus_v     = defocus_u.clone()
        defocus_angle = torch.zeros(N, dtype=torch.float32, device=device)
    else:
        # Non-astigmatic case with scalar defocus prior
        defocus_u     = defocus.view(-1) * 1e4     # µm → Å
        defocus_v     = defocus_u.clone()
        defocus_angle = torch.zeros(N, dtype=torch.float32, device=device)

    ctf = generate_ctf(
        num_pixels=num_pixels,
        pixel_size=pixel_size,
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        defocus_angle=defocus_angle,
        voltage=voltage,
        cs=cs,
        amplitude_contrast=amp.view(-1),
        b_factor=b_factor.view(-1),
        device=device,
    )

    return apply_ctf_batch(image, ctf)
