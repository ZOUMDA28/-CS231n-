import {
  BookOpen, ArrowRight, Check, Layers, Cpu, Star,
  Monitor, ChevronRight, CodeXml, Sparkles, Github,
  Image as ImageIcon, Target, Route, Zap, Brain, Network, Eye,
} from 'lucide-react'
import { GITHUB_OWNER, GITHUB_REPO, GITHUB_REPO_URL } from '../config.js'
import { PATH_STEPS, RUNNABLE_NOTEBOOKS } from '../data/sidebar.js'

const SECTION_STYLES = {
  'deep-learning-basics': {
    bg: 'from-[#dbeafe] to-[#e0f2fe]',
    tag: 'bg-blue-50 text-blue-600 border-blue-200/50',
    name: '深度学习',
    iconBg: 'bg-blue-100 text-blue-600 border-blue-200/50',
    pathBorder: 'border-l-blue-400',
    gradient: 'from-blue-500 to-cyan-500',
    soft: 'bg-blue-50 border-blue-200/50 text-blue-600',
  },
  'cnn-vision': {
    bg: 'from-[#ffedd5] to-[#fed7aa]',
    tag: 'bg-orange-50 text-orange-600 border-orange-200/50',
    name: 'CNN视觉',
    iconBg: 'bg-orange-100 text-orange-600 border-orange-200/50',
    pathBorder: 'border-l-orange-400',
    gradient: 'from-orange-500 to-amber-500',
    soft: 'bg-orange-50 border-orange-200/50 text-orange-600',
  },
  'frontiers': {
    bg: 'from-[#ede9fe] to-[#f5f3ff]',
    tag: 'bg-purple-50 text-purple-600 border-purple-200/50',
    name: '前沿',
    iconBg: 'bg-purple-100 text-purple-600 border-purple-200/50',
    pathBorder: 'border-l-purple-400',
    gradient: 'from-purple-500 to-violet-500',
    soft: 'bg-purple-50 border-purple-200/50 text-purple-600',
  },
}

const SECTION_ICONS = {
  'deep-learning-basics': Brain,
  'cnn-vision': Eye,
  'frontiers': Network,
}

export default function Welcome({ onSelect }) {
  const handleNotebookSelect = (nb) => {
    onSelect(nb.lessonId)
  }

  const handlePathSelect = (step) => {
    const firstNotebook = RUNNABLE_NOTEBOOKS.find((nb) => nb.section === step.section)
    if (firstNotebook) {
      handleNotebookSelect(firstNotebook)
    }
  }

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 sm:space-y-8 max-w-7xl w-full mx-auto">
      {/* HERO BANNER */}
      <section className="hero rounded-3xl p-6 md:p-10 relative overflow-hidden shadow-sm border bg-gradient-to-br from-[#eff6ff]/90 via-[#fff7ed] to-[#f5f3ff] border-blue-100/50">
        <div className="absolute top-[-20%] right-[-10%] w-[350px] h-[350px] rounded-full bg-blue-400/10 blur-[80px] pointer-events-none"></div>
        <div className="absolute bottom-[-10%] left-[20%] w-[250px] h-[250px] rounded-full bg-purple-300/10 blur-[60px] pointer-events-none"></div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold bg-[#e0f2fe] text-blue-600 border border-blue-200/50 shadow-sm">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse"></span>
              <span>Stanford CS231n 课程教程</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-[46px] font-extrabold tracking-tight text-slate-900 leading-[1.2]">
              Stanford CS231n
              <br className="hidden md:inline" />
              <span className="bg-gradient-to-r from-blue-600 via-orange-500 to-purple-600 bg-clip-text text-transparent">
                深度学习与计算机视觉
              </span>
            </h1>

            <p className="text-xs sm:text-sm md:text-base leading-relaxed text-slate-600 max-w-xl">
              基于斯坦福 CS231n 课程，通过可运行的 Notebook 系统学习计算机视觉与深度学习。涵盖从图像分类到 CNN、Transformer、检测分割、自监督学习与生成模型。
            </p>

            <div className="flex flex-wrap items-center gap-3 sm:gap-4 pt-2">
              <button
                onClick={() => {
                  const firstNb = RUNNABLE_NOTEBOOKS[0]
                  if (firstNb) handleNotebookSelect(firstNb)
                }}
                className="h-10 sm:h-12 px-5 sm:px-6 rounded-full bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs sm:text-sm shadow-lg shadow-blue-500/10 hover:shadow-blue-500/20 active:scale-[0.98] transition-all flex items-center gap-2"
              >
                <span>开始学习</span>
                <ArrowRight className="w-4 h-4" />
              </button>
              <a
                href={GITHUB_REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="h-10 sm:h-12 px-5 sm:px-6 rounded-full bg-white hover:bg-slate-50 text-slate-700 border border-slate-200/90 font-bold text-xs sm:text-sm active:scale-[0.98] transition-all flex items-center gap-2"
              >
                <Github className="w-4 h-4" />
                <span>GitHub</span>
              </a>
              <button
                onClick={() => {
                  const el = document.getElementById('learning-path-section')
                  if (el) el.scrollIntoView({ behavior: 'smooth' })
                }}
                className="h-10 sm:h-12 px-5 sm:px-6 rounded-full border border-blue-200/80 bg-white/85 hover:bg-white shadow-sm hover:shadow-md transition-all active:scale-[0.99] flex items-center gap-2"
              >
                <Route className="w-4 h-4 text-blue-600 shrink-0" />
                <span className="text-xs sm:text-sm font-extrabold text-slate-900 whitespace-nowrap">
                  浏览学习路径
                </span>
                <ChevronRight className="w-4 h-4 text-blue-600 group-hover:translate-x-0.5 transition-transform shrink-0" />
              </button>
            </div>

            <div className="flex items-start gap-3 max-w-xl rounded-2xl border border-blue-200/80 bg-white/75 px-3.5 py-3 shadow-sm">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
                <CodeXml className="h-4 w-4" strokeWidth={2.2} />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm font-extrabold text-slate-900 leading-snug">
                  每篇 Notebook 遵循直觉→手算→代码→实验四步教学路径
                </p>
                <p className="mt-1 text-[10px] sm:text-xs leading-relaxed text-slate-500">
                  浏览器内直接渲染，无需本地环境配置。核心算法全部从零实现，附带作业练习与论文引用。
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-x-4 sm:gap-x-6 gap-y-2 sm:gap-y-3 border-t border-slate-200/50 pt-4 sm:pt-5 max-w-lg select-none">
              {['可运行 Notebook', '学术论文引用', '从零实现算法', '作业与练习'].map((feature, idx) => (
                <div key={idx} className="flex items-center gap-2 text-[10px] sm:text-xs font-bold text-slate-500">
                  <div className="w-4 h-4 sm:w-4.5 sm:h-4.5 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 shrink-0">
                    <Check className="w-2.5 h-2.5 sm:w-3 sm:h-3 stroke-[2.5]" />
                  </div>
                  <span className="truncate">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="hidden lg:flex lg:col-span-5 relative min-h-[300px] items-center justify-center select-none">
            <div className="absolute w-[280px] h-[200px] rounded-2xl glass-effect shadow-xl p-4 border border-white/60 left-[5%] top-[5%] animate-float-1 z-10 overflow-hidden bg-white/80 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-3 border-b border-slate-200/40 pb-1.5">
                <div className="flex gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-green-400"></span>
                </div>
                <span className="text-[9px] font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">conv2d.py</span>
              </div>
              <pre className="font-mono text-[10px] text-slate-600 space-y-0.5">
                <div><span className="text-purple-600 font-bold">import</span> numpy <span className="text-purple-600 font-bold">as</span> np</div>
                <div className="text-slate-400"># 从零实现卷积</div>
                <div><span className="text-blue-600 font-bold">def</span> <span className="text-indigo-600 font-bold">conv2d</span>(x, kernel):</div>
                <div>  C_out, _, KH, KW = kernel.shape</div>
                <div>  H_out = (H - KH) // S + 1</div>
                <div>  out = np.<span className="text-purple-600">zeros</span>((C_out, H_out, W_out))</div>
                <div>  <span className="text-blue-600 font-bold">for</span> co <span className="text-blue-600 font-bold">in</span> <span className="text-purple-600">range</span>(C_out):</div>
                <div>    <span className="text-blue-600 font-bold">for</span> i <span className="text-blue-600 font-bold">in</span> <span className="text-purple-600">range</span>(H_out):</div>
                <div>      receptive = x[:, i*S:i*S+KH, ...]</div>
                <div>      out[co, i, j] = np.<span className="text-purple-600">sum</span>(receptive * kernel[co])</div>
              </pre>
            </div>

            <div className="absolute w-[200px] h-[150px] rounded-2xl glass-effect shadow-lg p-3.5 border border-white/60 right-0 bottom-[10%] animate-float-2 z-0 bg-white/80 backdrop-blur-sm">
              <div className="flex justify-between items-center text-[9px] font-semibold text-slate-500 mb-2">
                <span className="font-bold flex items-center gap-1">
                  <ImageIcon className="w-3 h-3 text-orange-500" />
                  CNN 特征图
                </span>
                <span>3x3 filters</span>
              </div>
              <div className="grid grid-cols-6 gap-0.5 pt-0.5">
                {[
                  'bg-orange-600/10', 'bg-orange-600/20', 'bg-orange-600/40', 'bg-orange-600/60', 'bg-orange-600/30', 'bg-orange-600/10',
                  'bg-orange-600/20', 'bg-orange-600/50', 'bg-orange-600/80', 'bg-orange-600/70', 'bg-orange-600/40', 'bg-orange-600/15',
                  'bg-orange-600/30', 'bg-orange-600/70', 'bg-orange-600/95', 'bg-orange-600/85', 'bg-orange-600/50', 'bg-orange-600/20',
                  'bg-orange-600/25', 'bg-orange-600/65', 'bg-orange-600/90', 'bg-orange-600/75', 'bg-orange-600/45', 'bg-orange-600/20',
                  'bg-orange-600/15', 'bg-orange-600/40', 'bg-orange-600/60', 'bg-orange-600/50', 'bg-orange-600/30', 'bg-orange-600/10',
                  'bg-orange-600/5', 'bg-orange-600/15', 'bg-orange-600/25', 'bg-orange-600/20', 'bg-orange-600/10', 'bg-orange-600/5',
                ].map((cls, j) => (
                  <div key={j} className={`h-4 rounded-sm ${cls}`}></div>
                ))}
              </div>
            </div>

            <div className="absolute top-[0%] right-[30%] bg-white/80 p-2.5 rounded-full shadow-md animate-float-3 border border-white/50 z-10">
              <Target className="w-4.5 h-4.5 text-amber-500" />
            </div>
            <div className="absolute bottom-[30%] left-[15%] bg-white/85 p-2 rounded-xl shadow-md animate-float-1 border border-white/50 z-20">
              <Sparkles className="w-4 h-4 text-purple-600" />
            </div>
          </div>
        </div>
      </section>

      {/* STATS BAR */}
      <section className="stats grid grid-cols-[repeat(auto-fit,minmax(min(100%,190px),1fr))] bg-white rounded-2xl border border-slate-200/70 shadow-sm overflow-hidden">
        <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-slate-50/45 transition-colors">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100/50 shrink-0">
            <BookOpen className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
          </div>
          <div className="space-y-0.5 min-w-0">
            <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-900 tracking-tight leading-none">10</div>
            <div className="text-[10px] sm:text-[11px] font-medium text-slate-500 leading-snug break-words">可运行 Notebook</div>
          </div>
        </div>
        <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-slate-50/45 transition-colors">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-orange-50 text-orange-600 flex items-center justify-center border border-orange-100/50 shrink-0">
            <Layers className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
          </div>
          <div className="space-y-0.5 min-w-0">
            <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-900 tracking-tight leading-none">3</div>
            <div className="text-[10px] sm:text-[11px] font-medium text-slate-500 leading-snug break-words">大学习路径</div>
          </div>
        </div>
        <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-slate-50/45 transition-colors">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center border border-emerald-100/50 shrink-0">
            <Cpu className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
          </div>
          <div className="space-y-0.5 min-w-0">
            <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-900 tracking-tight leading-none">20+</div>
            <div className="text-[10px] sm:text-[11px] font-medium text-slate-500 leading-snug break-words">核心算法</div>
          </div>
        </div>
        <div className="p-4 sm:p-5 md:p-6 flex items-center gap-3 sm:gap-4 hover:bg-slate-50/45 transition-colors">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center border border-amber-100/50 shrink-0">
            <Star className="w-5 h-5 sm:w-6 sm:h-6 stroke-[1.5]" />
          </div>
          <div className="space-y-0.5 min-w-0">
            <div className="text-lg sm:text-xl md:text-2xl font-bold text-slate-900 tracking-tight leading-none">CS231n</div>
            <div className="text-[10px] sm:text-[11px] font-medium text-slate-500 leading-snug break-words">Stanford 课程</div>
          </div>
        </div>
      </section>

      {/* FEATURES STRIP */}
      <section data-tour="features" className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,180px),1fr))] bg-white rounded-2xl border border-slate-200/70 shadow-sm p-3 sm:p-4 gap-3 sm:gap-4">
        {[
          { icon: <Monitor className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: '可运行 Notebook', desc: '浏览器内直接渲染，无需环境配置', iconClass: 'bg-blue-50 text-blue-600 border-blue-100/50' },
          { icon: <ImageIcon className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: '四步教学路径', desc: '直觉→手算→代码→实验', iconClass: 'bg-orange-50 text-orange-600 border-orange-100/50' },
          { icon: <Zap className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: '从零实现', desc: '核心算法不用封装库', iconClass: 'bg-amber-50 text-amber-600 border-amber-100/50' },
          { icon: <CodeXml className="w-4 h-4 sm:w-4.5 sm:h-4.5 stroke-[2]" />, title: '学术严谨', desc: '引用最新 arXiv 论文', iconClass: 'bg-emerald-50 text-emerald-600 border-emerald-100/50' },
        ].map((f, i) => (
          <div key={i} className="p-1.5 sm:p-2 flex items-start gap-2.5 sm:gap-3.5">
            <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border shrink-0 ${f.iconClass}`}>{f.icon}</div>
            <div className="space-y-0.5 min-w-0">
              <h3 className="text-[11px] sm:text-[13px] font-bold text-slate-900 leading-snug break-words">{f.title}</h3>
              <p className="text-[10px] sm:text-[11px] text-slate-500 font-medium leading-normal break-words">{f.desc}</p>
            </div>
          </div>
        ))}
      </section>

      {/* LEARNING PATH */}
      <section id="learning-path-section" className="parts bg-white rounded-2xl border border-slate-200/70 shadow-sm p-5 md:p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h2 className="text-[18px] md:text-[20px] font-bold text-slate-900">学习路径</h2>
            <p className="text-xs text-slate-500 font-medium">三大模块，逐步深入</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 relative">
          {PATH_STEPS.map((step, idx) => {
            const style = SECTION_STYLES[step.section] || SECTION_STYLES['deep-learning-basics']
            const Icon = SECTION_ICONS[step.section] || Brain
            return (
              <div key={idx} className="relative w-full">
                <div
                  onClick={() => handlePathSelect(step)}
                  className={`w-full rounded-[12px] p-4 sm:p-5 border border-slate-200/70 flex flex-col justify-between shadow-sm relative hover:shadow-md transition-all duration-200 cursor-pointer group overflow-hidden bg-gradient-to-br ${style.bg}`}
                >
                  <div className="flex items-start gap-3 sm:gap-4">
                    <div className={`w-12 h-12 sm:w-14 sm:h-14 rounded-xl bg-gradient-to-br ${style.gradient} flex items-center justify-center text-white shadow-lg shrink-0`}>
                      <Icon className="w-6 h-6 sm:w-7 sm:h-7" strokeWidth={2} />
                    </div>
                    <div className="space-y-1 flex-1 min-w-0">
                      <span className="text-[11px] font-mono font-bold tracking-wider px-2 py-0.5 rounded bg-white/60 text-slate-700 w-fit inline-block">
                        {step.num}
                      </span>
                      <h3 className="text-base sm:text-lg font-bold text-slate-900 group-hover:text-blue-700 transition-colors">
                        {step.title}
                      </h3>
                      <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                        {step.desc}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-500">
                      {RUNNABLE_NOTEBOOKS.filter(nb => nb.section === step.section).length} 个 Notebook
                    </span>
                    <div className="inline-flex items-center gap-1 text-sm font-medium group-hover:gap-2 transition-all text-slate-700 group-hover:text-blue-600">
                      开始学习
                      <ChevronRight className="w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* RUNNABLE NOTEBOOKS */}
      <section data-tour="notebooks" className="bg-white rounded-2xl border border-slate-200/70 shadow-sm p-5 md:p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h2 className="text-[18px] md:text-[20px] font-bold text-slate-900">精选 Notebook</h2>
            <p className="text-xs text-slate-500 font-medium">10 个核心章节，点击即可开始学习</p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3.5">
          {RUNNABLE_NOTEBOOKS.map((nb) => {
            const style = SECTION_STYLES[nb.section] || SECTION_STYLES['deep-learning-basics']
            const Icon = SECTION_ICONS[nb.section] || Brain
            return (
              <div
                key={nb.id}
                onClick={() => handleNotebookSelect(nb)}
                className="border border-slate-200/70 rounded-[10px] overflow-hidden cursor-pointer bg-white relative hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group flex flex-col"
              >
                <div className={`h-[90px] flex items-center justify-center relative overflow-hidden bg-gradient-to-br ${style.bg}`}>
                  <Icon className="w-8 h-8 opacity-60" strokeWidth={1.5} />
                </div>
                <div className="p-3 bg-white flex-1 flex flex-col justify-between">
                  <div>
                    <h4 className="text-[12px] font-semibold text-slate-900 group-hover:text-blue-600 transition-colors mb-1 line-clamp-1">{nb.title}</h4>
                    <p className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">{nb.desc}</p>
                  </div>
                  <div className="flex items-center justify-between text-[10px] mt-2">
                    <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold border ${style.tag}`}>
                      {style.name}
                    </span>
                    <span className="text-slate-500 font-medium">{nb.duration}分钟</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      {/* FOOTER */}
      <div className="w-full flex items-center justify-center pt-2 pb-6 select-none">
        <span className="text-xs text-slate-400 tracking-wide italic">
          Based on Stanford CS231n: Deep Learning for Computer Vision
        </span>
      </div>
    </div>
  )
}
