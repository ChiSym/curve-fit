import { assert } from "chai"
import { logpdf_normal } from "./stats"
import { describe, it  } from "mocha"

describe(`logpdf normal tests`, () => {
    it('works', () => {
        assert.equal(logpdf_normal(0.1, 0.0, 1.0), 0.2323)
    })
})
