<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/kai271828">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">Perutation Flowshop Scheduling using Simulated Annealing</h3>

  <p align="center">
    Implemented in Python
    <!-- <br />
    <a href="https://github.com/kai271828"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kai271828">View Demo</a>
    ·
    <a href="https://github.com/kai271828/.../issues">Report Bug</a>
    ·
    <a href="https://github.com/kai271828/.../issues">Request Feature</a>
  </p> -->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <!-- <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul> -->
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <!-- <li><a href="#prerequisites">Prerequisites</a></li> -->
        <!-- <li><a href="#installation">Installation</a></li> -->
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#search-minimum-makespan-with-specific-parameters">Search minimum makespan with specific parameters</a></li>
        <li><a href="#search-the-best-parameters-using-random-search">Search the best parameters using random search</a></li>
      </ul>
    </li>
    <!-- <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)-->

This repository is for the first homework assignment of the Meta-heuristics and Problem Solving course at NTNU in the spring semester of 2024.



<!-- ### Built With -->

<!-- This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples. -->

<!-- * [![Python][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

<!-- This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. -->

<!-- ### Prerequisites

Install packages through pip.
* pip
  ```sh
  pip install -r requirements.txt
  ``` -->

### Installation

Install packages through pip.

1. Clone the repo
   ```sh
   git clone https://github.com/kai271828/Permutation-Flowshop-Scheduling-using-Simulated-Annealing.git
   ```
2. Install packages
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Search minimum makespan with specific parameters
Here is an example:
```sh
python exp.py \
    --data_dir "data/tai20_5_1.txt" \
    --epoch_len 1 \
    --alpha 0.99 \
    --stopcriterion 1 \
    --temperature 1000 \
    --times 20 \
    --log_dir [PATH_TO_LOG_DIRECTORY]
```

|  Parameter   | Description  |
|  ----  | ----  |
| data_dir  | Path to the input file. |
| epoch_len  | Searching times before temperature decrease.  |
| alpha  | Cooling factor.  |
| stopcriterion  | Lowest temperature before stopping.  |
| temperature  | Initial searching temperature.  |
| times  | Total run times of this experiment. Related to output statistic. |
| log_dir  | Total run times of this experiment.  |
| verbose  | Weather to show the experiment detail. Default to False. |

### Search the best parameters using random search
Run the following command to try to find the optimal configuration of Simulated Annealing within the given range.
```sh
python search.py \
    --data_dir "data/tai20_5_1.txt" \
    --min_epoch_len 1 \
    --max_epoch_len 11 \
    --min_alpha 0.8 \
    --max_alpha 0.99 \
    --min_stopcriterion 1 \
    --max_stopcriterion 2 \
    --min_temperature 100 \
    --max_temperature 10000 \
    --times 20 \
    --search_times 10000 \
    --metric "avg"
```
Most parameters are quite similar to above except you should give a range [min, max).
|  Parameter   | Description  |
|  ----  | ----  |
| search_times  | The number of configurations searched. |
| metric  | How to evaluate the result. {"best", "avg", "worst"}, default to "avg"  |

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!-- ## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/kai271828/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE -->
<!-- ## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
<!-- ## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/kai271828/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/kai271828/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kai271828/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/kai271828/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/kai271828/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/kai271828/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/kai271828/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/kai271828/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/kai271828/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/kai271828/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kai271828
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com  -->
