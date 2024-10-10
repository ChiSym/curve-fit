//import {jest} from '@jest/globals'
import {test, expect} from '@jest/globals'
import { logpdf_normal, logpdf_uniform } from './stats'

test(`logpdf normal tests`, () => {
    expect(logpdf_normal(0.1, 0.0, 1.0)).toBeCloseTo(-0.923939)
    expect(logpdf_uniform(0.1, -1.0, 1.0)).toBeCloseTo(-0.693147)
})
