# Self-Speculative Decoding

The experiment results of `gemma-2-9b-it` as follows:

```
Generate token number: 102
Generate speed: 24.07665693982178 token/sec
Speculative Decoding Spent Time: 4.236468553543091 seconds.
Accept Rate: 0.4056603773584906

Generate token number: 100
Generate speed: 29.647649721089135 token/sec
Normal Target Model Decoding Spent Time: 3.37294864654541 seconds.

Generate token number: 100
Generate speed: 48.81880782264111 token/sec
Normal Draft Model Decoding Spent Time: 2.0483908653259277 seconds.
```

It's not really good, but I think the continue tasks can improve it.