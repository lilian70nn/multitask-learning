# Vendored frontend libraries

Served locally so the app does not load scripts from third-party CDNs. All are
MIT licensed. Downloaded on 2026-11-27; do not edit the files.

| File | Package | Version | Source URL | Licence |
|---|---|---|---|---|
| `react-18.3.1.production.min.js` | react | 18.3.1 | https://unpkg.com/react@18.3.1/umd/react.production.min.js | MIT, (c) Facebook, Inc. and its affiliates |
| `react-dom-18.3.1.production.min.js` | react-dom | 18.3.1 | https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js | MIT, (c) Facebook, Inc. and its affiliates |
| `babel-standalone-7.29.15.min.js` | @babel/standalone | 7.29.15 | https://unpkg.com/@babel/standalone@7.29.15/babel.min.js | MIT, (c) 2014-present Sebastian McKenzie and other contributors |
| `tailwindcss-play-3.4.17.js` | tailwindcss (Play CDN build) | 3.4.17 | https://cdn.tailwindcss.com/3.4.17 | MIT, (c) Tailwind Labs, Inc. |

Babel is pinned to major version 7: version 8 changes the default JSX runtime,
which breaks in-browser `<script type="text/babel">` with the React UMD builds.

## SHA-256

```
b2ae47b8c770a13659caae47c258fa2661153197f56fc74ec7d182c5d7331825  babel-standalone-7.29.15.min.js
d949f1c3687aedadcedac85261865f29b17cd273997e7f6b2bfc53b2f9d4c4dd  react-18.3.1.production.min.js
35f4f974f4b2bcd44da73963347f8952e341f83909e4498227d4e26b98f66f0d  react-dom-18.3.1.production.min.js
176e894661aa9cdc9a5cba6c720044cbbf7b8bd80d1c9a142a7c24b1b6c50d15  tailwindcss-play-3.4.17.js
```

## MIT licence text

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
