#set document(
  title: [Is General Artificial Intelligence Improvement Sustainable?],
)

#title()

= Introduction
From 2019 to 2024, global private investment in general artificial intelligence
ballooned from around one billion dollars to almost 34 billion dollars. An
underlying thesis of continued investment is (a) that those investments will
produce returns (i.e., that general artificial intelligence companies will
continue to grow in values and either dividends will pay out or the investor
will be able to sell their shares in the company for a higher value than they
paid) and (b) that those returns are significant enough to justify the initial
cost of investment. Although global private investment has grown exponentially
and the valuations of artificial intelligence companies like OpenAI or Anthropic
have grown similarly exponentially over time, we argue that, on a basic
technical level, this growth is unsustainable. General artificial intelligence
requires compute to train: compute which requires microprocessors and money to
purchase them. We draw a parallel to Moore's law, which is the thesis that the
number of transistors per microprocessor follows an exponential curve. Moore's
law couldn't have happened if transistors didn't decrease in price during its
growth period. Transistors are to microprocessors what microprocessors are to
compute. Even if investments continue at the pace they have---which is itself
unlikely and unsustainable---more and more compute continues to be devoted to
general artificial intelligence models with fewer and fewer returns per petaflop
of compute. Compute demand vastly outpaces transistor price decline. Therefore,
we argue that the current compute demands of general artificial intelligence are
unsustainable, contributing to the hypothesis that general artificial
intelligence investment itself is unsustainable.
= Training Computation Over Time
@tc1 shows training computation over time, starting in 1971. @tc2 shows training
computation over time, beginning in 2011.

#grid(
  columns: (1fr, 1fr),
  column-gutter: 10pt,
  [#figure(
    scale(x: 50%, y: 50%, reflow: true,
      image("figs\fig1_1.png")),
    caption: [Training computation in petaFLOPS on a log scale against time
    starting in 1971]
  ) <tc1>],
  [#figure(
    scale(x: 50%, y: 50%, reflow: true,
      image("figs\fig1_2.png")),
    caption: [Training computation in petaFLOPS on a log scale against time
    starting in 2011]
  ) <tc2>],
)

Both graphs demonstrate a clear exponential correlation between time and
training computation.
