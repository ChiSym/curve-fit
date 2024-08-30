import type { LC, PropsWithChildren } from '@use-gpu/live'
import { AutoCanvas, WebGPU } from '@use-gpu/webgpu'
import { Cartesian, Axis, Grid } from '@use-gpu/plot'

type ComponentProps = {
    canvas: HTMLCanvasElement
}

export const Component: LC<ComponentProps> = (props: PropsWithChildren<ComponentProps>) => {
    const {canvas} = props;
    return (
        <WebGPU fallback={(error: Error) => <span className="error">{error.message}</span>}>
            <AutoCanvas canvas={canvas}>
                <Cartesian range={[[-1, 1], [-1, 1]]}>
                    <Axis axis="x"></Axis>
                    <Axis axis="y" />
                    <Grid axes="x,y" />
                </Cartesian>
            </AutoCanvas>
        </WebGPU>
    )

    //console.log('got here', canvas)
}
