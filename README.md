![](docs/images/teaser.png)

# Falcor

Falcor is a real-time rendering framework supporting DirectX 12 and Vulkan. It aims to improve productivity of research and prototype projects.

Custom Features include:
- Importance Resampling for Global Illumination (EGSR 2005, Talbot)
- ReSTIR DI (SIGG, 2020, Bitterli) (but It's not implemented for Alias Table)

## Resources
- [Falcor](https://github.com/NVIDIAGameWorks/Falcor): Falcor's GitHub page.
- [Documentation](./docs/index.md): Additional information and tutorials.
    - [Getting Started](./docs/getting-started.md)
    - [Render Graph Tutorials](./docs/tutorials/index.md)
- [Rendering Resources](https://benedikt-bitterli.me/resources) A collection of scenes loadable in Falcor (pbrt-v4 format).
- [ORCA](https://developer.nvidia.com/orca): A collection of scenes and assets optimized for Falcor.
- [Slang](https://github.com/shader-slang/slang): Falcor's shading language and compiler.

## Citation
If you use Falcor in a research project leading to a publication, please cite the project.
The BibTex entry is

```bibtex
@Misc{Kallweit22,
   author =      {Simon Kallweit and Petrik Clarberg and Craig Kolb and Tom{'a}{\v s} Davidovi{\v c} and Kai-Hwa Yao and Theresa Foley and Yong He and Lifan Wu and Lucy Chen and Tomas Akenine-M{\"o}ller and Chris Wyman and Cyril Crassin and Nir Benty},
   title =       {The {Falcor} Rendering Framework},
   year =        {2022},
   month =       {8},
   url =         {https://github.com/NVIDIAGameWorks/Falcor},
   note =        {\url{https://github.com/NVIDIAGameWorks/Falcor}}
}
```
